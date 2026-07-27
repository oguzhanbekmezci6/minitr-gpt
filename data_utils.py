from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CorpusSplits:
    train: torch.Tensor
    validation: torch.Tensor


def load_text(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {source}")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Veri dosyası boş.")
    return text


def encode_and_split(
    token_ids: list[int],
    train_ratio: float = 0.90,
) -> CorpusSplits:
    if not 0.5 <= train_ratio < 1.0:
        raise ValueError("train_ratio [0.5, 1.0) aralığında olmalıdır.")
    if len(token_ids) < 100:
        raise ValueError("Korpus çok küçük; en az 100 token gereklidir.")

    data = torch.tensor(token_ids, dtype=torch.long)
    split_index = int(len(data) * train_ratio)
    return CorpusSplits(
        train=data[:split_index],
        validation=data[split_index:],
    )


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Veri bölümü ({len(data)} token), block_size + 1 "
            f"({block_size + 1}) değerinden büyük olmalıdır."
        )

    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
