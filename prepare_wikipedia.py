from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests


API_URL = "https://tr.wikipedia.org/w/api.php"
URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
MULTISPACE_PATTERN = re.compile(r"[ \t]+")
MANY_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Türkçe Wikipedia API'sinden temiz metin korpusu hazırla"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/turkish_wikipedia.txt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/wikipedia_manifest.jsonl"),
    )
    parser.add_argument("--target-mb", type=float, default=10.0)
    parser.add_argument("--batch-pages", type=int, default=10)
    parser.add_argument("--min-article-chars", type=int, default=700)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--user-agent",
        type=str,
        default=os.environ.get("WIKIMEDIA_USER_AGENT", ""),
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = URL_PATTERN.sub("", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = MULTISPACE_PATTERN.sub(" ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if line.startswith(("Kategori:", "Dosya:", "Şablon:")):
            continue
        if len(line) < 2:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = MANY_NEWLINES_PATTERN.sub("\n\n", text)
    return text.strip()


def fetch_random_pages(
    session: requests.Session,
    batch_pages: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(batch_pages),
        "prop": "extracts|info",
        "explaintext": "1",
        "inprop": "url",
        "redirects": "1",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(API_URL, params=params, timeout=45)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload.get("query", {}).get("pages", [])
        except (requests.RequestException, ValueError, RuntimeError) as error:
            last_error = error
            sleep_seconds = min(2 ** attempt, 16)
            print(f"API hatası; {sleep_seconds} sn sonra tekrar deneniyor: {error}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Wikipedia API isteği başarısız: {last_error}")


def main() -> None:
    args = parse_args()
    if args.target_mb <= 0:
        raise ValueError("target-mb pozitif olmalıdır.")
    if not 1 <= args.batch_pages <= 20:
        raise ValueError("batch-pages 1 ile 20 arasında olmalıdır.")
    if not args.user_agent.strip():
        raise ValueError(
            "Açıklayıcı bir User-Agent gereklidir. --user-agent ver veya "
            "WIKIMEDIA_USER_AGENT ortam değişkenini ayarla."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_mb * 1024 * 1024)
    written_bytes = args.output.stat().st_size if args.output.exists() else 0
    known_page_ids: set[int] = set()
    known_hashes: set[str] = set()

    if args.manifest.exists():
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                known_page_ids.add(int(record["pageid"]))
                if "content_sha256" in record:
                    known_hashes.add(record["content_sha256"])
            except (ValueError, KeyError, json.JSONDecodeError):
                continue

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent,
            "Accept": "application/json",
        }
    )

    accepted = 0
    skipped = 0
    print(
        f"Başlangıç: {written_bytes / 1024 / 1024:.2f} MB | "
        f"Hedef: {args.target_mb:.2f} MB"
    )

    with args.output.open("a", encoding="utf-8") as corpus_handle, args.manifest.open(
        "a", encoding="utf-8"
    ) as manifest_handle:
        while written_bytes < target_bytes:
            pages = fetch_random_pages(
                session,
                batch_pages=args.batch_pages,
                max_retries=args.max_retries,
            )

            for page in pages:
                page_id = page.get("pageid")
                title = str(page.get("title", "")).strip()
                extract = str(page.get("extract", ""))

                if not page_id or page_id in known_page_ids:
                    skipped += 1
                    continue
                if title.lower().endswith("(anlam ayrımı)"):
                    skipped += 1
                    continue

                cleaned = clean_text(extract)
                if len(cleaned) < args.min_article_chars:
                    skipped += 1
                    known_page_ids.add(int(page_id))
                    continue

                content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                if content_hash in known_hashes:
                    skipped += 1
                    known_page_ids.add(int(page_id))
                    continue

                record_text = cleaned + "\n\n"
                corpus_handle.write(record_text)
                corpus_handle.flush()

                manifest_record = {
                    "pageid": int(page_id),
                    "title": title,
                    "url": page.get("fullurl"),
                    "retrieved_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "characters": len(cleaned),
                    "content_sha256": content_hash,
                    "license_note": "Wikipedia text; verify CC BY-SA/GFDL attribution requirements.",
                }
                manifest_handle.write(
                    json.dumps(manifest_record, ensure_ascii=False) + "\n"
                )
                manifest_handle.flush()

                added_bytes = len(record_text.encode("utf-8"))
                written_bytes += added_bytes
                accepted += 1
                known_page_ids.add(int(page_id))
                known_hashes.add(content_hash)

                print(
                    f"{written_bytes / 1024 / 1024:7.2f}/{args.target_mb:.2f} MB | "
                    f"kabul={accepted} atlandı={skipped} | {title}"
                )
                if written_bytes >= target_bytes:
                    break

            time.sleep(max(args.delay, 0.0))

    print(f"Korpus hazır: {args.output}")
    print(f"Kaynak manifesti: {args.manifest}")


if __name__ == "__main__":
    main()
