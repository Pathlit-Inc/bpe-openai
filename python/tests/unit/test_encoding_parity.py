from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import bpe_openai as candidate

FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "performance" / "fixtures"
)


def load_prompt() -> str:
    return (FIXTURES_DIR / "long_form_prompt.txt").read_text(encoding="utf-8")


try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - tiktoken not available/offline
    tiktoken = None


@pytest.mark.skipif(tiktoken is None, reason="tiktoken not available for parity check")
def test_encoding_matches_tiktoken() -> None:
    text = load_prompt()

    try:
        tk_encoding = tiktoken.encoding_for_model("gpt-4o")
    except Exception as exc:  # pragma: no cover - offline environment
        pytest.skip(f"tiktoken unavailable: {exc}")

    candidate_encoding = candidate.encoding_for_model("gpt-4o")
    expected_tokens = tk_encoding.encode(text)

    assert candidate_encoding.encode(text) == expected_tokens
    assert candidate_encoding.decode(expected_tokens) == tk_encoding.decode(expected_tokens)
