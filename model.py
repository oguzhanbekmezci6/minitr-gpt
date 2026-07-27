from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Q, K ve V projeksiyonları elle yazılmış masked multi-head attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        self.qkv_projection = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.output_projection = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, channels = x.shape
        if channels != self.n_embd:
            raise ValueError("Beklenmeyen embedding boyutu.")

        qkv = self.qkv_projection(x)
        query, key, value = qkv.split(self.n_embd, dim=2)

        query = query.view(
            batch_size, sequence_length, self.n_head, self.head_size
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.n_head, self.head_size
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.n_head, self.head_size
        ).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        dropped_weights = self.attention_dropout(attention_weights)

        output = dropped_weights @ value
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        output = self.residual_dropout(self.output_projection(output))

        returned_weights = attention_weights if return_attention else None
        return output, returned_weights


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_size, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer bloğu."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_output, weights = self.attention(
            self.layer_norm_1(x),
            return_attention=return_attention,
        )
        x = x + attention_output
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x, weights


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_layer_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.language_model_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # Token embedding ve çıktı ağırlıklarını paylaşmak parametre sayısını azaltır.
        self.language_model_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)
        self._scale_residual_projections()

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("output_projection.weight"):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    def parameter_count(self, exclude_position_embedding: bool = True) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if exclude_position_embedding:
            total -= self.position_embedding.weight.numel()
        return total

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Dizi uzunluğu {sequence_length}, block_size {self.config.block_size} değerini aşıyor."
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)
        x = self.embedding_dropout(token_vectors + position_vectors)

        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x, return_attention=return_attention)
            if weights is not None:
                attention_maps.append(weights)

        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss, attention_maps if return_attention else None

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens negatif olamaz.")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalıdır.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")

        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                effective_k = min(top_k, next_token_logits.size(-1))
                threshold = torch.topk(next_token_logits, effective_k).values[:, -1]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold.unsqueeze(-1),
                    float("-inf"),
                )

            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids
