# PyMind-A: Independent AI Development Policy

## Goal

PyMind-A develops and trains its own model weights using architectures implemented in this repository. It must not require a third-party hosted AI inference API to train or run its core models.

## Independence rules

- Do not call Gemini, Hugging Face Inference, OpenAI, Anthropic, or other hosted model-generation APIs.
- Do not download or load pretrained model checkpoints for the independent model track.
- Initialize model parameters from random values in our own model implementation.
- Train using datasets that are public-domain, openly licensed for the intended use, or created by the project owner.
- Record dataset source, license, preprocessing, model configuration, seed, and evaluation results.
- Keep optional non-AI software libraries separate from model weights and inference services.
- The website is a control panel; it is not itself the model and must not fabricate model responses.

## What "from scratch" means here

The project writes its own model architecture and starts training from randomly initialized weights. It may use general-purpose numerical and deep-learning libraries such as PyTorch to perform tensor operations and automatic differentiation. Using PyTorch does not mean using a pretrained AI model.

## Current baseline

The repository already contains a compact decoder-only Transformer (pymind/model.py), a UTF-8 byte tokenizer, a training loop, checkpoint saving, evaluation, and generation. The independent foundation package added in Phase 1 provides a small validated configuration, a reproducible random-initialization factory, and a synthetic training smoke test.

## Scope and limitations

The initial model is an educational, small CPU-friendly experiment. It is not a frontier-scale assistant. Training quality depends on dataset size and quality, model capacity, compute, and careful evaluation. Image and video generation require separate research tracks and significantly more compute.

## Reproducibility

Run the foundation smoke test:

    python -m pymind.independent

Run its tests:

    python -m pytest tests/test_independent_foundation.py -q
