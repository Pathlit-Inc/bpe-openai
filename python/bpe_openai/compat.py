from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class ModelMetadata:
    encoding: str
    chunk_limit: int
    special_tokens: Mapping[str, int]


_DEFAULT_SPECIALS = {"<|endoftext|>": 100_257, "<|reserved_special_0|>": 100_258}

MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    "cl100k_base": ModelMetadata(
        encoding="cl100k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "o200k_base": ModelMetadata(
        encoding="o200k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "voyage3_base": ModelMetadata(
        encoding="voyage3_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "gpt-4o": ModelMetadata(
        encoding="o200k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "gpt-4o-mini": ModelMetadata(
        encoding="o200k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "gpt-4.1": ModelMetadata(
        encoding="o200k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "gpt-4.1-mini": ModelMetadata(
        encoding="o200k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "gpt-4o-128k": ModelMetadata(
        encoding="o200k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "gpt-4.1-128k": ModelMetadata(
        encoding="o200k_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
    "voyage-3": ModelMetadata(
        encoding="voyage3_base",
        chunk_limit=200_000,
        special_tokens=_DEFAULT_SPECIALS,
    ),
}


def list_supported_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def get_metadata(model_name: str) -> ModelMetadata:
    key = model_name.lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - handled by higher-level error
        raise KeyError(model_name) from exc


def get_default_special_tokens(encoding_name: str) -> Mapping[str, int]:
    metadata = MODEL_REGISTRY.get(encoding_name.lower())
    if metadata:
        return metadata.special_tokens
    # Fallback to primary model metadata if encoding maps elsewhere.
    for item in MODEL_REGISTRY.values():
        if item.encoding == encoding_name:
            return item.special_tokens
    return {}
