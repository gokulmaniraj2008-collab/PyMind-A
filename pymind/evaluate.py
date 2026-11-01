"""Evaluate a byte-tokenizer PyMind-A checkpoint on a held-out UTF-8 text file."""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from pymind.model import GPTConfig, TinyGPT
from pymind.tokenizer import ByteTokenizer


@torch.no_grad()
def evaluate(model, token_ids, block_size, device):
    if len(token_ids) < 2:
        raise ValueError("Evaluation text must contain at least two UTF-8 bytes.")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        # Each window predicts the following bytes; windows do not overlap.
        for start in range(0, len(token_ids) - 1, block_size):
            window = token_ids[start:start + block_size + 1]
            if len(window) < 2:
                continue
            x = torch.tensor(window[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(window[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                y.reshape(-1),
                reduction="sum",
            )
            total_loss += loss_sum.item()
            total_tokens += y.numel()
    finally:
        model.train(was_training)
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 80.0)),
            "tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Held-out UTF-8 evaluation text")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    if checkpoint.get("tokenizer", {}).get("type") != ByteTokenizer.tokenizer_type:
        parser.error("This evaluator currently supports UTF-8 byte-tokenizer checkpoints only.")
    config = GPTConfig(**checkpoint["config"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    token_ids = ByteTokenizer().encode(Path(args.data).read_text(encoding="utf-8"))
    try:
        metrics = evaluate(model, token_ids, config.block_size, device)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Evaluation bytes: {metrics['tokens']}")
    print(f"Next-byte loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print("Note: perplexity is a dataset-specific metric, not a direct measure of answer quality.")


if __name__ == "__main__":
    main()
