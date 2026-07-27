from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable


UNKNOWN_SYMBOL = "�"


class CharacterTokenizer:
    """Basit, açıklanabilir karakter seviyeli tokenizer."""

    def __init__(self, vocabulary: Iterable[str]) -> None:
        chars = list(vocabulary)
        if not chars:
            raise ValueError("Vocabulary boş olamaz.")
        if len(set(chars)) != len(chars):
            raise ValueError("Vocabulary içinde tekrarlı karakterler var.")

        if UNKNOWN_SYMBOL not in chars:
            chars.append(UNKNOWN_SYMBOL)

        self.chars = chars
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = {index: char for index, char in enumerate(chars)}
        self.unk_id = self.stoi[UNKNOWN_SYMBOL]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        normalized = unicodedata.normalize("NFC", text)
        return cls(sorted(set(normalized)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        normalized = unicodedata.normalize("NFC", text)
        return [self.stoi.get(char, self.unk_id) for char in normalized]

    def decode(self, token_ids: Iterable[int]) -> str:
        output: list[str] = []
        for token_id in token_ids:
            index = int(token_id)
            output.append(self.itos.get(index, UNKNOWN_SYMBOL))
        return "".join(output)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "character",
            "normalization": "NFC",
            "unknown_symbol": UNKNOWN_SYMBOL,
            "vocabulary": self.chars,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "character":
            raise ValueError("Desteklenmeyen tokenizer türü.")
        return cls(payload["vocabulary"])
