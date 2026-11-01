import torch

from pymind.model import GPTConfig, TinyGPT


def small_model():
    config = GPTConfig(
        vocab_size=20,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )
    return TinyGPT(config)


def test_forward_shapes_and_loss():
    model = small_model()
    tokens = torch.randint(0, 20, (2, 12))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 12, 20)
    assert loss is not None
    assert torch.isfinite(loss)


def test_generation_appends_tokens():
    model = small_model()
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=5)
    assert generated.shape == (1, 8)


def test_causal_mask_blocks_future_positions():
    model = small_model()
    mask = model.blocks[0].attention.causal_mask[0, 0]
    assert torch.all(mask.triu(1) == 0)
    assert torch.all(mask.diagonal() == 1)
