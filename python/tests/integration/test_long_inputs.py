from __future__ import annotations

import pytest

import bpe_openai as candidate


def test_long_inputs_trigger_chunk_limit_error() -> None:
    encoding = candidate.encoding_for_model("gpt-4o")
    long_text = " token" * 250_000

    with pytest.raises(candidate.TokenLimitError) as excinfo:  # type: ignore[attr-defined]
        encoding.encode(long_text)

    assert "chunk limit" in str(excinfo.value).lower()
    assert "250000" in str(excinfo.value)
