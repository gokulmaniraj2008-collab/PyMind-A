"""Build and smoke-test PyMind-A models without loading pretrained weights."""

import math

import torch

from pymind.independent.config import IndependentConfig
from pymind.model import TinyGPT


MODEL_ORIGIN = "random_initialization"


def create_fresh_model(
    config: IndependentConfig | None = None,
    seed: int = 1337,
) -> TinyGPT:
    """Create a new TinyGPT from random weights without changing global RNG state.

    This function never downloads a checkpoint, contacts a model provider, or loads
    pretrained parameters. The seed makes initialization reproducible.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    config = config or IndependentConfig()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyGPT(config.to_gpt_config())

    model.pymind_model_origin = MODEL_ORIGIN
    model.pymind_seed = seed
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_smoke_test(
    config: IndependentConfig | None = None,
    seed: int = 1337,
) -> dict:
    """Run one synthetic next-token training step and report finite loss/gradients."""
    config = config or IndependentConfig()
    model = create_fresh_model(config=config, seed=seed)
    model.train()

    sequence_length = min(config.block_size, 8)
    tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, sequence_length + 1),
    )
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(inputs, targets)

    if loss is None or not math.isfinite(loss.item()):
        raise RuntimeError("model produced a non-finite training loss")

    loss.backward()
    gradient_tensors = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradient_tensors:
        raise RuntimeError("no gradients were produced")
    if not all(torch.isfinite(gradient).all().item() for gradient in gradient_tensors):
        raise RuntimeError("model produced non-finite gradients")

    optimizer.step()
    return {
        "model_origin": MODEL_ORIGIN,
        "seed": seed,
        "parameters": count_parameters(model),
        "initial_loss": float(loss.item()),
        "gradient_tensors": len(gradient_tensors),
        "training_step": "passed",
    }
