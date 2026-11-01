"""A compact decoder-only GPT-style Transformer implemented in PyTorch."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size < 1 or self.n_layer < 1:
            raise ValueError("block_size and n_layer must be positive")

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        batch, length, channels = x.shape
        key = self.key(x).view(batch, length, self.n_head, self.head_size).transpose(1, 2)
        query = self.query(x).view(batch, length, self.n_head, self.head_size).transpose(1, 2)
        value = self.value(x).view(batch, length, self.n_head, self.head_size).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) * (self.head_size ** -0.5)
        scores = scores.masked_fill(
            self.causal_mask[:, :, :length, :length] == 0,
            float("-inf"),
        )
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.language_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch, length = input_ids.shape
        if length > self.config.block_size:
            raise ValueError("input sequence exceeds configured block_size")

        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.language_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size),
                targets.reshape(batch * length),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        repetition_penalty=1.0,
    ):
        """Sample tokens with temperature, top-k/top-p and repetition controls."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.block_size:]
                logits, _ = self(context)
                next_logits = logits[:, -1, :].clone()

                if repetition_penalty > 1:
                    for row in range(input_ids.size(0)):
                        seen = input_ids[row].unique()
                        seen_logits = next_logits[row, seen]
                        next_logits[row, seen] = torch.where(
                            seen_logits < 0,
                            seen_logits * repetition_penalty,
                            seen_logits / repetition_penalty,
                        )

                next_logits = next_logits / temperature

                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    threshold = torch.topk(next_logits, k, dim=-1).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative_probs > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    filtered = torch.full_like(next_logits, float("-inf"))
                    next_logits = filtered.scatter(1, sorted_indices, sorted_logits)

                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
        finally:
            if was_training:
                self.train()
        return input_ids
