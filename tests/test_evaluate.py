import torch

from pymind.evaluate import evaluate
from pymind.model import GPTConfig, TinyGPT


def test_evaluate_reports_finite_loss_and_restores_training_mode():
    config = GPTConfig(vocab_size=256, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    model.train()
    result = evaluate(model, list("hello world".encode("utf-8")), 8, "cpu")
    assert result["tokens"] == len("hello world".encode("utf-8")) - 1
    assert result["loss"] > 0
    assert result["perplexity"] >= 1
    assert model.training
