import torch

from pymind.model import GPTConfig, TinyGPT
from pymind.train import estimate_loss, get_batch


def test_get_batch_returns_shifted_sequences():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_rejects_short_data():
    try:
        get_batch(torch.arange(8), block_size=8, batch_size=2, device="cpu")
    except ValueError as exc:
        assert "more than 8 tokens" in str(exc)
    else:
        raise AssertionError("Expected short dataset to be rejected")


def test_estimate_loss_returns_finite_metrics_and_restores_mode():
    config = GPTConfig(vocab_size=16, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    model.train()
    data = torch.randint(0, 16, (100,))
    metrics = estimate_loss(model, data[:80], data[80:], config, 2, "cpu", batches=2)
    assert set(metrics) == {"train", "val"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert model.training
