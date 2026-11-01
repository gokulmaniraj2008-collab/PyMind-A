# PyMind-A

**A Python-first playground for building and understanding language models.**

PyMind-A is an educational decoder-only GPT project built with Python and PyTorch. It learns next-token prediction from the UTF-8 text datasets you provide. It is a foundation for experimenting with language models—not a ChatGPT replacement or a ready-made image/video generator.

## Current capabilities

- Decoder-only Transformer with masked multi-head self-attention
- UTF-8 byte tokenizer (256-token vocabulary), supporting Tamil, English, emoji, and other Unicode text
- Configurable dataset path and model/training parameters
- Train/validation loss metrics exported to CSV
- Reproducible initialization with a configurable random seed
- CPU support; CUDA is selected automatically when available
- Save and reload checkpoints; Phase 1 character-tokenizer checkpoints remain supported
- Unit tests and GitHub Actions smoke tests

## Project structure

```text
PyMind-A/
├── data/
│   └── input.txt
├── logs/                  # generated CSV metrics (git-ignored)
├── checkpoints/           # generated model weights (git-ignored)
├── pymind/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
├── tests/
├── .github/workflows/
├── requirements.txt
└── README.md
```

## Quick start

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

Train with the included sample dataset:

```bash
python -m pymind.train --data data/input.txt --max-iters 1000
```

Train on your own UTF-8 text file with a smaller model suitable for a CPU smoke run:

```bash
python -m pymind.train --data data/my_text.txt --max-iters 100 --block-size 32 --n-embd 32 --n-head 4 --n-layer 1 --seed 42
```

Generate text:

```bash
python -m pymind.generate --checkpoint checkpoints/tiny_gpt.pt --prompt "வணக்கம்"
```

Run tests:

```bash
pytest -q
```

## Training outputs and reproducibility

By default, training writes:

- Model checkpoint: `checkpoints/tiny_gpt.pt`
- Loss history: `logs/training_metrics.csv`

The CSV contains `step`, `train_loss`, and `val_loss`. Use `--metrics-output` and `--output` to change their locations. Use `--seed` to reproduce the random initialization and sampling sequence on the same software/hardware setup; exact results can still vary across hardware and library versions.

Useful options:

| Option | Default | Purpose |
|---|---:|---|
| `--data` | `data/input.txt` | UTF-8 training dataset |
| `--max-iters` | `1000` | Training steps |
| `--eval-interval` | `100` | Loss measurement interval |
| `--batch-size` | `32` | Sequences per batch |
| `--block-size` | `64` | Context length in bytes |
| `--learning-rate` | `0.0003` | AdamW learning rate |
| `--seed` | `1337` | Random seed |

## How it works

1. The tokenizer converts UTF-8 text into byte IDs from 0 to 255.
2. Token and position embeddings create input representations.
3. Causal self-attention lets each position attend only to itself and earlier positions.
4. Feed-forward layers transform each position.
5. The output head predicts the next byte.
6. Training minimizes next-byte cross-entropy loss.

Because this is a small model trained on a tiny sample, generated text is experimental and may not be meaningful. Better results require a larger, clean dataset and more compute. Do not upload private data, secrets, or copyrighted datasets unless you have the right to use them.

## Roadmap

- Phase 1: Tiny GPT foundation
- Phase 2: UTF-8 tokenizer, configurable data, training metrics, reproducibility
- Phase 3: Model evaluation, improved sampling, and larger configurations
- Later: Optional API/service layer and carefully scoped integrations

## License

No license has been added yet. Add a license before accepting external contributions or distributing the project under specific reuse terms.
