"""Generate text from a saved PyMind-A checkpoint."""

import argparse
from pathlib import Path

import torch

from pymind.model import GPTConfig, TinyGPT
from pymind.tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    parser.add_argument("--prompt", default="PyMind")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40,
                        help="Keep only the K most likely next tokens; use 0 to disable.")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold in (0, 1].")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens cannot be negative")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if not args.prompt:
        parser.error("--prompt cannot be empty")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the interval (0, 1]")
    if args.repetition_penalty < 1:
        parser.error("--repetition-penalty must be >= 1")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        Path(args.checkpoint), map_location=device, weights_only=False
    )
    config = GPTConfig(**checkpoint["config"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if checkpoint.get("tokenizer", {}).get("type") == ByteTokenizer.tokenizer_type:
        tokenizer = ByteTokenizer()
        prompt_ids = tokenizer.encode(args.prompt)
        decode = tokenizer.decode
    else:
        # Backward compatibility with Phase 1 character-tokenizer checkpoints.
        stoi = checkpoint["stoi"]
        itos = {int(index): character for index, character in checkpoint["itos"].items()}
        unknown = sorted(set(args.prompt) - set(stoi))
        if unknown:
            parser.error(f"Prompt contains characters not in model vocabulary: {unknown}")
        prompt_ids = [stoi[character] for character in args.prompt]
        decode = lambda ids: "".join(itos[token] for token in ids)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )[0].tolist()
    print(decode(output))


if __name__ == "__main__":
    main()
