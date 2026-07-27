from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Mini decoder-only Transformer yapılandırması."""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size pozitif olmalıdır.")
        if self.block_size <= 0:
            raise ValueError("block_size pozitif olmalıdır.")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("Katman, head ve embedding boyutları pozitif olmalıdır.")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd, n_head değerine tam bölünmelidir.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout [0, 1) aralığında olmalıdır.")
