from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import GPTConfig
from model import MiniGPT
from tokenizer import CharacterTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention ağırlıklarını incele")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def printable(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == "\t":
        return "\\t"
    if char == " ":
        return "[boşluk]"
    return char


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer_path = args.tokenizer or (
        args.checkpoint.parent / checkpoint["tokenizer_file"]
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    config = GPTConfig(**checkpoint["model_config"])

    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    token_ids = tokenizer.encode(args.text)[-config.block_size :]
    if not token_ids:
        raise ValueError("İncelenecek metin boş olamaz.")

    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    _, _, attention_maps = model(input_tensor, return_attention=True)
    assert attention_maps is not None

    layer_index = args.layer if args.layer >= 0 else len(attention_maps) - 1
    if not 0 <= layer_index < len(attention_maps):
        raise IndexError("Geçersiz layer numarası.")
    if not 0 <= args.head < config.n_head:
        raise IndexError("Geçersiz attention head numarası.")

    # [batch, head, query_position, key_position]
    weights = attention_maps[layer_index][0, args.head, -1]
    characters = [tokenizer.decode([token_id]) for token_id in token_ids]
    ranked = sorted(
        zip(characters, weights.tolist(), range(len(characters))),
        key=lambda item: item[1],
        reverse=True,
    )[: args.top]

    print(
        f"Katman: {layer_index} | Head: {args.head} | "
        f"Sorgu karakteri: {printable(characters[-1])}"
    )
    print("-" * 54)
    print(f"{'Konum':>6}  {'Karakter':<14}  {'Attention':>10}")
    print("-" * 54)
    for character, weight, position in ranked:
        print(f"{position:>6}  {printable(character):<14}  {weight:>10.6f}")


if __name__ == "__main__":
    main()
