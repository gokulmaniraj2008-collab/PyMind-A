"""Train the Phase 1 character-level TinyGPT model."""

import argparse
from pathlib import Path

import torch

from pymind.model import GPTConfig, TinyGPT


def build_vocabulary(text):
    characters = sorted(set(text))
    stoi = {character: index for index, character in enumerate(characters)}
    itos = {index: character for character, index in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    unknown = sorted(set(text) - set(stoi))
    if unknown:
        raise ValueError(f"Text contains characters outside the training vocabulary: {unknown}")
    return [stoi[character] for character in text]


def get_batch(data, block_size, batch_size, device):
    if len(data) <= block_size:
        raise ValueError(
            f"Training split needs more than {block_size} characters; "
            "add more text or reduce --block-size."
        )
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, batch_size, device, batches=10):
    model.eval()
    results = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, config.block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--output", default="checkpoints/tiny_gpt.pt")
    args = parser.parse_args()

    if args.max_iters < 1 or args.eval_interval < 1 or args.batch_size < 1:
        parser.error("iteration, evaluation interval and batch size must be positive")

    text = Path(args.data).read_text(encoding="utf-8")
    stoi, itos = build_vocabulary(text)
    token_ids = torch.tensor(encode(text, stoi), dtype=torch.long)

    split = int(0.9 * len(token_ids))
    train_data, val_data = token_ids[:split], token_ids[split:]
    if len(train_data) <= args.block_size or len(val_data) <= args.block_size:
        parser.error(
            "Dataset is too short for the selected block size and 90/10 split. "
            "Add more text or lower --block-size."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device} | Vocabulary: {len(stoi)} | Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, config, args.batch_size, device
            )
            print(f"step {step:5d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

        x, y = get_batch(train_data, config.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "config": config.to_dict(),
            "stoi": stoi,
            "itos": itos,
        },
        output,
    )
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
