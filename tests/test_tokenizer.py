from pathlib import Path

from tokenizer import CharacterTokenizer


def test_roundtrip_known_characters(tmp_path: Path) -> None:
    text = "İstatistik ve yapay zekâ.\n"
    tokenizer = CharacterTokenizer.from_text(text)
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded) == text

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.vocab_size == tokenizer.vocab_size


def test_unknown_character_is_safe() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    encoded = tokenizer.encode("abç")
    assert len(encoded) == 3
    assert tokenizer.decode(encoded).endswith("�")
