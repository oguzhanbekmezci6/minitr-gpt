from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from config import GPTConfig
from data_utils import CorpusSplits, encode_and_split, get_batch, load_text
from model import MiniGPT
from tokenizer import CharacterTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniTR-GPT eğitim betiği")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/run"))
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Desteklenen sistemlerde torch.compile kullanır.",
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default="Bilim",
        help="Değerlendirme sırasında kısa örnek üretmek için başlangıç metni.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = [
        "max_iters",
        "eval_interval",
        "eval_iters",
        "log_interval",
        "batch_size",
        "gradient_accumulation_steps",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} pozitif olmalıdır.")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("Öğrenme oranları geçersiz.")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate, learning-rate değerinden büyük olamaz.")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA istendi ancak kullanılabilir değil.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS istendi ancak kullanılabilir değil.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_for_step(
    step: int,
    warmup_iters: int,
    max_iters: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if warmup_iters > 0 and step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
    if step >= max_iters:
        return min_lr
    if max_iters <= warmup_iters:
        return min_lr

    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    splits: CorpusSplits,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for split_name, split_data in (
        ("train", splits.train),
        ("validation", splits.validation),
    ):
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            x, y = get_batch(split_data, batch_size, block_size, device)
            _, loss, _ = model(x, y)
            assert loss is not None
            losses[index] = loss.detach().cpu()
        results[split_name] = losses.mean().item()
    model.train()
    return results


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    step: int,
    best_validation_loss: float,
    tokenizer_path: Path,
    args: argparse.Namespace,
) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(config),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "tokenizer_file": tokenizer_path.name,
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{step}.tmp")
    torch.save(checkpoint, temporary_path)

    # Windows bazen checkpoint dosyasını kısa süreliğine kilitliyor.
    # Bu durumda dosya değiştirme işlemi birkaç kez yeniden denenir.
    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            temporary_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.15 * (attempt + 1))

    # Kilit devam ederse adım numaralı ayrı bir checkpoint kaydedilir.
    fallback_path = path.with_name(f"{path.stem}-step-{step}{path.suffix}")
    temporary_path.replace(fallback_path)
    print(
        f"Uyarı: {path.name} Windows tarafından kilitlendi; "
        f"checkpoint {fallback_path.name} olarak kaydedildi. Hata: {last_error}"
    )


def append_metric(path: Path, payload: dict[str, float | int]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = choose_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.out_dir / "tokenizer.json"
    metrics_path = args.out_dir / "metrics.jsonl"

    text = load_text(args.data)

    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_tokenizer = args.resume.parent / checkpoint["tokenizer_file"]
        tokenizer = CharacterTokenizer.load(resume_tokenizer)
        config = GPTConfig(**checkpoint["model_config"])
        if config.block_size != args.block_size:
            print(
                f"Not: resume checkpoint block_size={config.block_size}; "
                f"komut satırındaki {args.block_size} kullanılmayacak."
            )
    else:
        tokenizer = CharacterTokenizer.from_text(text)
        tokenizer.save(tokenizer_path)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )

    tokenizer.save(tokenizer_path)
    splits = encode_and_split(tokenizer.encode(text), train_ratio=args.train_ratio)
    if len(splits.validation) <= config.block_size:
        raise ValueError(
            "Validation bölümü block_size değerinden küçük. Daha büyük veri veya daha küçük block_size kullan."
        )

    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    best_validation_loss = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("Bu PyTorch sürümünde torch.compile yok.")
        model = torch.compile(model)

    training_config = {
        "model": asdict(config),
        "data_file": str(args.data),
        "device": str(device),
        "train_tokens": len(splits.train),
        "validation_tokens": len(splits.validation),
        "vocab_size": tokenizer.vocab_size,
        "parameters": (
            model._orig_mod.parameter_count() if hasattr(model, "_orig_mod") else model.parameter_count()
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "training_config.json").write_text(
        json.dumps(training_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Cihaz: {device}")
    print(f"Vocabulary: {tokenizer.vocab_size} karakter")
    print(f"Train token: {len(splits.train):,}")
    print(f"Validation token: {len(splits.validation):,}")
    print(f"Model parametresi: {training_config['parameters']:,}")
    print(f"Başlangıç adımı: {start_step}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started_at = time.perf_counter()
    progress = tqdm(range(start_step, args.max_iters), desc="Eğitim", unit="step")

    for step in progress:
        current_lr = learning_rate_for_step(
            step,
            args.warmup_iters,
            args.max_iters,
            args.learning_rate,
            args.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(
                splits.train,
                args.batch_size,
                config.block_size,
                device,
            )
            _, loss, _ = model(x, y)
            assert loss is not None
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += loss.detach().item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            progress.set_postfix(
                loss=f"{accumulated_loss / args.gradient_accumulation_steps:.4f}",
                lr=f"{current_lr:.2e}",
            )

        should_evaluate = step % args.eval_interval == 0 or step == args.max_iters - 1
        if should_evaluate:
            losses = estimate_loss(
                model,
                splits,
                args.eval_iters,
                args.batch_size,
                config.block_size,
                device,
            )
            elapsed = time.perf_counter() - started_at
            metric = {
                "step": step,
                "train_loss": losses["train"],
                "validation_loss": losses["validation"],
                "learning_rate": current_lr,
                "elapsed_seconds": elapsed,
            }
            append_metric(metrics_path, metric)
            print(
                f"\nAdım {step}: train={losses['train']:.4f}, "
                f"validation={losses['validation']:.4f}"
            )

            is_best = losses["validation"] < best_validation_loss
            if is_best:
                best_validation_loss = losses["validation"]

            save_checkpoint(
                args.out_dir / "last.pt",
                model,
                optimizer,
                config,
                step,
                best_validation_loss,
                tokenizer_path,
                args,
            )

            if is_best:
                save_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    optimizer,
                    config,
                    step,
                    best_validation_loss,
                    tokenizer_path,
                    args,
                )
                print(f"Yeni en iyi model kaydedildi: val_loss={best_validation_loss:.4f}")

            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            prompt_ids = tokenizer.encode(args.sample_prompt)
            if prompt_ids:
                prompt_tensor = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=device
                )
                generated = raw_model.generate(
                    prompt_tensor,
                    max_new_tokens=120,
                    temperature=0.8,
                    top_k=min(30, tokenizer.vocab_size),
                )
                print("Örnek çıktı:")
                print(tokenizer.decode(generated[0].tolist()))

    print(f"Eğitim tamamlandı. Çıktı klasörü: {args.out_dir}")


if __name__ == "__main__":
    main()
