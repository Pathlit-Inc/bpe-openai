from __future__ import annotations

from pathlib import Path

import pytest

import bpe_openai as candidate


FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "performance" / "fixtures"


try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


def load_prompt() -> str:
    return (FIXTURES / "long_form_prompt.txt").read_text(encoding="utf-8")


@pytest.mark.skipif(tiktoken is None, reason="tiktoken not available")
def test_existing_script_runs_without_code_changes() -> None:
    text = load_prompt()

    try:
        baseline = tiktoken.encoding_for_model("gpt-4o")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiktoken unavailable: {exc}")

    expected_tokens = baseline.encode(text)
    candidate_tokens = candidate.encoding_for_model("gpt-4o").encode(text)

    assert candidate_tokens == expected_tokens
