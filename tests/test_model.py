import pytest
import torch

from pymind.model import GPTConfig, TinyGPT
from pymind.tokenizer import ByteTokenizer


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


def test_generation_supports_sampling_controls():
    model = small_model()
    prompt = torch.tensor([[1, 2, 3]])
    torch.manual_seed(7)
    first = model.generate(prompt.clone(), max_new_tokens=6, top_k=5,
                           top_p=0.8, repetition_penalty=1.2)
    torch.manual_seed(7)
    second = model.generate(prompt.clone(), max_new_tokens=6, top_k=5,
                            top_p=0.8, repetition_penalty=1.2)
    assert torch.equal(first, second)
    assert first.shape == (1, 9)


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0},
    {"top_k": 0},
    {"top_p": 0},
    {"top_p": 1.1},
    {"repetition_penalty": 0.9},
    {"max_new_tokens": -1},
])
def test_generation_rejects_invalid_sampling_parameters(kwargs):
    with pytest.raises(ValueError):
        small_model().generate(torch.tensor([[1, 2]]), **kwargs)


def test_generation_restores_training_mode():
    model = small_model()
    model.train()
    model.generate(torch.tensor([[1, 2]]), max_new_tokens=1)
    assert model.training


def test_causal_mask_blocks_future_positions():
    model = small_model()
    mask = model.blocks[0].attention.causal_mask[0, 0]
    assert torch.all(mask.triu(1) == 0)
    assert torch.all(mask.diagonal() == 1)


def test_byte_tokenizer_round_trips_unicode():
    tokenizer = ByteTokenizer()
    text = "PyMind-A தமிழ் 🚀"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == 256


def test_byte_tokenizer_rejects_non_string():
    with pytest.raises(TypeError):
        ByteTokenizer().encode(123)
