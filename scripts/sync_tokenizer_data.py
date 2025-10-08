#!/usr/bin/env python3
"""Copy tokenizer data files into the Python package for packaging builds."""

from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "vendor" / "rust-gems" / "crates" / "bpe-openai" / "data"
TARGET_DIR = REPO_ROOT / "python" / "bpe_openai" / "data"


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Missing tokenizer data directory: {SOURCE_DIR}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    init_file = TARGET_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    for path in SOURCE_DIR.glob("*.tiktoken.gz"):
        shutil.copy2(path, TARGET_DIR / path.name)


if __name__ == "__main__":
    main()
