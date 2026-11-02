"""Tests for PyMind-A's independent model foundation."""

import pytest
import torch

from pymind.independent import IndependentConfig, create_fresh_model, run_smoke_test
from pymind.independent.factory import MODEL_ORIGIN, count_parameters


def test_default_config_is_small_and_valid():
    config = IndependentConfig()
    assert config.vocab_size == 256
    assert config.block_size == 32
    assert config.n_layer == 1
    assert config.to_gpt_config().n_embd == 32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0},
        {"block_size": 0},
        {"n_embd": 30, "n_head": 4},
        {"dropout": 1.0},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        IndependentConfig(**kwargs)


def test_same_seed_reproduces_random_initialization():
    first = create_fresh_model(seed=17)
    second = create_fresh_model(seed=17)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_different_seed_changes_initial_weights():
    first = create_fresh_model(seed=17)
    second = create_fresh_model(seed=18)
    assert any(
        not torch.equal(value, second.state_dict()[name])
        for name, value in first.state_dict().items()
    )


def test_factory_marks_model_as_randomly_initialized():
    model = create_fresh_model(seed=7)
    assert model.pymind_model_origin == MODEL_ORIGIN
    assert model.pymind_seed == 7
    assert count_parameters(model) > 0


def test_smoke_test_performs_finite_training_step():
    result = run_smoke_test(seed=1337)
    assert result["model_origin"] == "random_initialization"
    assert result["parameters"] > 0
    assert result["gradient_tensors"] > 0
    assert result["training_step"] == "passed"
    assert result["initial_loss"] > 0
