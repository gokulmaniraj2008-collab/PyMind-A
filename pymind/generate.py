"""Generate text from a saved PyMind-A checkpoint."""

import argparse
from pathlib import Path

import torch

from pymind.model import GPTConfig, TinyGPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    parser.add_argument("--prompt", default="PyMind")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens cannot be negative")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        Path(args.checkpoint), map_location=device, weights_only=False
    )
    config = GPTConfig(**checkpoint["config"])
    stoi = checkpoint["stoi"]
    itos = {int(index): character for index, character in checkpoint["itos"].items()}

    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        parser.error(f"Prompt contains characters not in model vocabulary: {unknown}")

    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_ids = torch.tensor(
        [[stoi[character] for character in args.prompt]],
        dtype=torch.long,
        device=device,
    )
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )[0].tolist()
    print("".join(itos[token] for token in output))


if __name__ == "__main__":
    main()
