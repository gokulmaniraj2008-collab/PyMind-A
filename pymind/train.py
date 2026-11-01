"""Train PyMind-A with validation, best-checkpoint saving and efficient updates."""

import argparse
import csv
import random
from pathlib import Path

import torch

from pymind.model import GPTConfig, TinyGPT
from pymind.tokenizer import ByteTokenizer


def get_batch(data, block_size, batch_size, device):
    if len(data) <= block_size:
        raise ValueError(
            f"Dataset split needs more than {block_size} tokens; "
            "add more text or reduce --block-size."
        )
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, batch_size, device, batches=10):
    """Estimate mean train/validation loss without changing the model's mode."""
    was_training = model.training
    model.eval()
    results = {}
    try:
        for name, data in (("train", train_data), ("val", val_data)):
            losses = []
            for _ in range(batches):
                x, y = get_batch(data, config.block_size, batch_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            results[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return results


def save_checkpoint(path, model, config, tokenizer, seed, step, val_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "seed": seed,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many evaluations without validation improvement; 0 disables.")
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="checkpoints/tiny_gpt.pt")
    parser.add_argument("--metrics-output", default="logs/training_metrics.csv")
    args = parser.parse_args()

    if min(args.max_iters, args.eval_interval, args.eval_batches,
           args.batch_size, args.grad_accum_steps) < 1:
        parser.error("iterations, evaluation settings, batch size and accumulation steps must be positive")
    if args.block_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.seed < 0:
        parser.error("block size and seed must be non-negative; learning rate must be positive and weight decay non-negative")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    text = Path(args.data).read_text(encoding="utf-8")
    if not text.strip():
        parser.error("Dataset is empty; add UTF-8 text to the data file.")
    tokenizer = ByteTokenizer()
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(0.9 * len(token_ids))
    train_data, val_data = token_ids[:split], token_ids[split:]
    if len(train_data) <= args.block_size or len(val_data) <= args.block_size:
        parser.error(
            "Dataset is too short for the selected block size and 90/10 split. "
            "Add more text or lower --block-size."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    evaluations_without_improvement = 0
    best_step = -1

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["step", "train_loss", "val_loss", "best_val_loss"],
        )
        writer.writeheader()
        print(
            f"Device: {device} | Tokenizer: UTF-8 bytes | "
            f"Vocabulary: {tokenizer.vocab_size} | "
            f"Parameters: {sum(p.numel() for p in model.parameters()):,} | Seed: {args.seed}"
        )

        for step in range(args.max_iters):
            if step % args.eval_interval == 0 or step == args.max_iters - 1:
                losses = estimate_loss(
                    model, train_data, val_data, config, args.batch_size,
                    device, batches=args.eval_batches,
                )
                improved = losses["val"] < best_val_loss
                if improved:
                    best_val_loss = losses["val"]
                    best_step = step
                    evaluations_without_improvement = 0
                    save_checkpoint(
                        args.output, model, config, tokenizer, args.seed,
                        step, best_val_loss,
                    )
                else:
                    evaluations_without_improvement += 1

                writer.writerow({
                    "step": step,
                    "train_loss": f"{losses['train']:.6f}",
                    "val_loss": f"{losses['val']:.6f}",
                    "best_val_loss": f"{best_val_loss:.6f}",
                })
                metrics_file.flush()
                print(
                    f"step {step:5d}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}, best val {best_val_loss:.4f}"
                )

                if (args.early_stopping_patience > 0
                        and evaluations_without_improvement >= args.early_stopping_patience):
                    print(f"Early stopping at step {step}; best checkpoint is from step {best_step}.")
                    break

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(args.grad_accum_steps):
                x, y = get_batch(train_data, config.block_size, args.batch_size, device)
                _, loss = model(x, y)
                (loss / args.grad_accum_steps).backward()
                accumulated_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    print(f"Best checkpoint: {args.output} (step {best_step}, val loss {best_val_loss:.4f})")
    print(f"Saved training metrics: {metrics_path}")


if __name__ == "__main__":
    main()
