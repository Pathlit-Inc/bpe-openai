from __future__ import annotations

import pytest

import bpe_openai as candidate


def test_long_inputs_trigger_chunk_limit_error() -> None:
    encoding = candidate.encoding_for_model("gpt-4o")
    # Use a string that exceeds the default 200k chunk limit without tripping the
    # defensive 1e6-character guard present in both tiktoken and this shim. CJK
    # characters are encoded one token per character, so this reliably exceeds the
    # limit while keeping the raw length well below 1e6.
    long_text = "一" * 210_000

    with pytest.raises(candidate.TokenLimitError) as excinfo:  # type: ignore[attr-defined]
        encoding.encode(long_text)

    assert "chunk limit" in str(excinfo.value).lower()
    assert "210000" in str(excinfo.value)
