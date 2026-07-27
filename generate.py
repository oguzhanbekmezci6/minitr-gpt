from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import GPTConfig
from model import MiniGPT
from tokenizer import CharacterTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniTR-GPT ile metin üret")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default="Türkiye")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer_path = args.tokenizer or (
        args.checkpoint.parent / checkpoint["tokenizer_file"]
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)

    config = GPTConfig(**checkpoint["model_config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt boş olamaz.")

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
