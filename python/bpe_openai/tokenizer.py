from __future__ import annotations

import importlib
from time import perf_counter
from typing import Iterable, List, Mapping, Optional, Sequence, Set

from . import compat, errors
from .configuration import TokenizerConfiguration, TokenizerRuntime
from .metrics import MetricsPayload, dispatch
from .results import TokenizationResult


def _load_backend():
    return importlib.import_module("bpe_openai._bindings")


class Encoding:
    """High-level tokenizer API exposed to Python callers."""

    def __init__(self, runtime: TokenizerRuntime, model: str) -> None:
        self._runtime = runtime
        self._model = model
        bindings = _load_backend()
        self._backend = bindings.tokenizer_for_model(model)
        self._backend_version = getattr(bindings, "RUST_BACKEND_VERSION", "unknown")
        self._last_result: Optional[TokenizationResult] = None
        self._metrics_hook = None

    def encode(
        self,
        text: str,
        *,
        allowed_special: Optional[Sequence[str]] = None,
        disallowed_special: Optional[Sequence[str]] = None,
    ) -> List[int]:
        self._validate_special_arguments(allowed_special, disallowed_special)

        token_count = self._backend.count(text)
        if token_count > self._runtime.chunk_limit:
            errors.raise_token_limit(token_count, self._runtime.chunk_limit)

        start = perf_counter()
        token_ids = self._backend.encode(text)
        elapsed_ms = (perf_counter() - start) * 1_000

        result = TokenizationResult(
            token_ids=token_ids,
            token_strings=[],
            total_tokens=len(token_ids),
            truncated=False,
            elapsed_ms=elapsed_ms,
        )
        self._last_result = result

        dispatch(
            self._metrics_hook,
            MetricsPayload(
                model=self._model,
                total_tokens=len(token_ids),
                elapsed_ms=elapsed_ms,
                rust_backend_version=self._backend_version,
            ),
        )

        return list(token_ids)

    def decode(self, token_ids: Iterable[int]) -> str:
        ids_list = list(token_ids)
        return self._backend.decode(ids_list)

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, batches: Sequence[Sequence[int]]) -> List[str]:
        return [self.decode(batch) for batch in batches]

    def register_special_tokens(self, mapping: Mapping[str, int]) -> None:
        raise NotImplementedError(
            "Custom special tokens are not supported; the underlying tokenizer is fixed."
        )

    def set_metrics_hook(self, callback) -> None:
        self._metrics_hook = callback
        self._runtime.set_metrics_hook(callback)

    @property
    def last_result(self) -> Optional[TokenizationResult]:
        return self._last_result

    def _validate_special_arguments(
        self,
        allowed_special: Optional[Sequence[str]],
        disallowed_special: Optional[Sequence[str]],
    ) -> None:
        if allowed_special and len(list(allowed_special)) > 0:
            raise NotImplementedError("allowed_special overrides are not supported yet")

        if disallowed_special and len(list(disallowed_special)) > 0:
            raise NotImplementedError("disallowed_special overrides are not supported yet")


def build_encoding_from_model(model_name: str) -> Encoding:
    config = TokenizerConfiguration.for_model(model_name)
    runtime = TokenizerRuntime(config)
    return Encoding(runtime, model=config.model_name)


def build_encoding_from_name(
    encoding_name: str,
    *,
    allowed_special: Optional[Sequence[str]] = None,
    disallowed_special: Optional[Sequence[str]] = None,
) -> Encoding:
    if allowed_special and len(list(allowed_special)) > 0:
        raise NotImplementedError("allowed_special overrides are not supported yet")
    if disallowed_special and len(list(disallowed_special)) > 0:
        raise NotImplementedError("disallowed_special overrides are not supported yet")

    default_specials = dict(compat.get_default_special_tokens(encoding_name))
    config = TokenizerConfiguration(
        model_name=encoding_name,
        encoding=encoding_name,
        special_tokens=default_specials,
    )
    config.validate()
    runtime = TokenizerRuntime(config)
    return Encoding(runtime, model=config.model_name)
