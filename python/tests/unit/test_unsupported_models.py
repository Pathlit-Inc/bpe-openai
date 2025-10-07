from __future__ import annotations

import pytest

import bpe_openai as candidate


def test_unsupported_model_surfaces_guidance() -> None:
    with pytest.raises(candidate.UnsupportedModelError) as excinfo:  # type: ignore[attr-defined]
        candidate.encoding_for_model("gpt-unknown-999")

    message = str(excinfo.value)
    assert "supported" in message.lower()
    assert "gpt-4o" in message or "cl100k" in message
