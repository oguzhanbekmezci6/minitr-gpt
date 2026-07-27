"""PyCharm'da parametre girmeden çalıştırılabilen küçük demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "demo" / "best.pt"


def run(command: list[str]) -> None:
    print("\n>", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run(
        [
            sys.executable,
            "train.py",
            "--data",
            "data/sample_turkish.txt",
            "--out-dir",
            "checkpoints/demo",
            "--max-iters",
            "300",
            "--eval-interval",
            "50",
            "--eval-iters",
            "10",
            "--batch-size",
            "16",
            "--block-size",
            "64",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "64",
        ]
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint oluşmadı: {CHECKPOINT}")

    run(
        [
            sys.executable,
            "generate.py",
            "--checkpoint",
            str(CHECKPOINT.relative_to(ROOT)),
            "--prompt",
            "Bilim",
            "--max-new-tokens",
            "300",
            "--temperature",
            "0.8",
            "--top-k",
            "20",
        ]
    )


if __name__ == "__main__":
    main()
