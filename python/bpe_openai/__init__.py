"""Tiktoken-compatible wrapper exposing the bpe-openai Rust tokenizer crate."""

from __future__ import annotations

from importlib import metadata
from typing import Iterable, Optional, Sequence

from . import compat
from .errors import (
    SpecialTokenCollisionError,
    TokenLimitError,
    TokenizerError,
    UnsupportedModelError,
)
from .tokenizer import Encoding, build_encoding_from_model, build_encoding_from_name

try:  # pragma: no cover - package metadata unavailable during local dev
    __version__ = metadata.version("bpe-openai")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "Encoding",
    "encoding_for_model",
    "get_encoding",
    "list_supported_models",
    "TokenizerError",
    "UnsupportedModelError",
    "SpecialTokenCollisionError",
    "TokenLimitError",
    "__version__",
]


def list_supported_models() -> list[str]:
    return compat.list_supported_models()


def _normalize_special_arg(value: Optional[Sequence[str] | str]) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "all":
            return None
        return [value]
    return value


def encoding_for_model(model_name: str) -> Encoding:
    return build_encoding_from_model(model_name)


def get_encoding(
    encoding_name: str,
    *,
    allowed_special: Optional[Sequence[str] | str] = None,
    disallowed_special: Optional[Sequence[str] | str] = None,
) -> Encoding:
    allowed = _normalize_special_arg(allowed_special)
    disallowed = _normalize_special_arg(disallowed_special)
    return build_encoding_from_name(
        encoding_name,
        allowed_special=allowed,
        disallowed_special=disallowed,
    )
