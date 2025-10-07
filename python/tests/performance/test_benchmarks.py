from __future__ import annotations

import time
from pathlib import Path

import bpe_openai as candidate
import pytest


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def read_prompt() -> str:
    return (FIXTURES / "long_form_prompt.txt").read_text(encoding="utf-8")


try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


def test_candidate_latency_matches_baseline_within_tolerance() -> None:
    text = read_prompt()

    if tiktoken is None:  # pragma: no cover - environment fallback
        pytest.skip("tiktoken not available; cannot run benchmark parity test")

    try:
        baseline_tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiktoken unavailable: {exc}")

    encoding = candidate.encoding_for_model("gpt-4o")

    start = time.perf_counter()
    candidate_tokens = encoding.encode(text)
    elapsed_ms = (time.perf_counter() - start) * 1_000

    assert candidate_tokens == baseline_tokens
    assert elapsed_ms < 50, f"Expected encode() under 50ms, saw {elapsed_ms:.2f}ms"
