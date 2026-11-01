# PyMind-A

**A Python-first playground for building and understanding language models.**

Phase 1 provides a small, trainable, decoder-only GPT-style Transformer implemented with Python and PyTorch. It includes a character tokenizer, causal self-attention, a training loop, checkpoint saving, text generation, and basic tests.

> This is an educational starter model, not a ChatGPT replacement. It learns only from the text you provide and will produce low-quality text when trained on the tiny demo corpus.

## Features

- Decoder-only Transformer with masked multi-head self-attention
- Character-level tokenizer built from the training text
- Train/validation split and loss reporting
- CPU support; CUDA is used automatically when available
- Save and reload model checkpoints
- Generate text from a prompt
- Small pytest model-shape and causal-mask tests

## Project structure

```text
PyMind-A/
├── data/
│   └── input.txt
├── pymind/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
├── tests/
│   └── test_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick start

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

Train the model:

```bash
python -m pymind.train --data data/input.txt --max-iters 1000
```

Generate text from a prompt:

```bash
python -m pymind.generate --checkpoint checkpoints/tiny_gpt.pt --prompt "PyMind"
```

Run tests:

```bash
pytest
```

For a quick CPU smoke test, use fewer iterations:

```bash
python -m pymind.train --max-iters 20 --eval-interval 10
```

## Add your own training data

Replace or extend `data/input.txt` with plain UTF-8 text. The model's vocabulary is built from the characters found in that file. A larger and cleaner dataset generally produces better results, but training also requires more compute and memory.

## Model design

1. Characters are mapped to integer token IDs.
2. Token and position embeddings create input representations.
3. Causal self-attention lets each position attend only to itself and earlier positions.
4. Feed-forward layers transform each position.
5. A linear output head predicts the next character.
6. Training minimizes next-character cross-entropy loss.

## Checkpoints and safety

Checkpoints are written to `checkpoints/`, which is ignored by Git. Do not commit private datasets, API keys, personal information, or large model weights. This Phase 1 project does not call external AI APIs and does not download pretrained model weights.

## Roadmap

- Phase 1: Tiny GPT foundation (current)
- Phase 2: Better tokenizer, configurable datasets, training metrics and reproducible experiments
- Phase 3: Larger model configurations and checkpoint evaluation
- Phase 4: Optional API/service layer and integrations

## License

No license has been added yet. Add a license before accepting external contributions or distributing the project under specific reuse terms.
