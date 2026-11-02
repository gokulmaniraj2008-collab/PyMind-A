# Phase 1 — Independent AI Foundation

## Deliverables

- A dedicated pymind.independent package that uses the repository's own TinyGPT implementation.
- A validated, small default model configuration for CPU experiments.
- A reproducible factory that creates a new model from random weights.
- A smoke test that executes a forward pass, computes next-token loss, checks gradients, and performs one optimizer step.
- Unit tests for configuration validation, reproducible initialization, seed separation, parameter counting, and training.
- An explicit independence policy covering model APIs, pretrained weights, dataset licensing, and honest capability claims.

## Commands

    python -m pymind.independent
    python -m pytest tests/test_independent_foundation.py -q

The existing pymind.model, pymind.train, pymind.generate, website builder, and Vercel files remain unchanged.

## Acceptance criteria

1. No third-party model API is needed to run the new foundation.
2. The factory constructs TinyGPT using random initialization only.
3. The same seed reproduces the same initial state on the same software/device setup.
4. Different seeds produce different initial weights.
5. A synthetic next-token training step produces finite loss and gradients.
6. Existing project tests and CI continue to run.
