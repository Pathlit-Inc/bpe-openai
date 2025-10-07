from __future__ import annotations

import pytest

import bpe_openai as candidate


def test_register_special_tokens_not_supported() -> None:
    encoding = candidate.get_encoding("cl100k_base")

    with pytest.raises(NotImplementedError):
        encoding.register_special_tokens({"<|endoftext|>": 1})
