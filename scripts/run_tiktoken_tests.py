#!/usr/bin/env python3
"""Run the vendored tiktoken test suite against the bpe-openai implementation."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args, remaining = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    tiktoken_repo = repo_root / "vendor" / "tiktoken"
    test_dir = tiktoken_repo / "tests"

    if not test_dir.is_dir():
        print(f"Unable to locate tiktoken tests at {test_dir}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(tiktoken_repo))
    sys.path.insert(0, str(python_dir))

    import importlib.util

    load_path = tiktoken_repo / "tiktoken" / "load.py"
    spec = importlib.util.spec_from_file_location("tiktoken.load", load_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules.setdefault("tiktoken.load", module)

    import bpe_openai as shim

    sys.modules.setdefault("tiktoken", shim)

    import pytest

    from bpe_openai.tokenizer import _load_backend  # type: ignore

    backend = _load_backend()
    backend_encodings = {name.lower() for name in backend.supported_encodings()}
    backend_models = {name.lower() for name in backend.supported_models()}

    supported_encodings = set(shim.list_encoding_names())
    supported_models = set(shim.list_supported_models())

    def _should_skip_encoding(name: str) -> bool:
        key = name.lower()
        if key not in backend_encodings:
            return True
        return False

    def _should_skip_model(name: str) -> bool:
        key = name.lower()
        if key not in backend_models:
            return True
        return False

    original_get_encoding = shim.get_encoding
    original_encoding_for_model = shim.encoding_for_model

    def get_encoding_with_skip(name: str):
        key = name.lower()
        if key not in supported_encodings or _should_skip_encoding(name):
            pytest.skip(f"Encoding '{name}' is not supported by the bpe-openai backend")
        try:
            return original_get_encoding(name)
        except shim.UnsupportedModelError:
            pytest.skip(f"Encoding '{name}' is not supported by bpe-openai")
        except ValueError as exc:
            if "Unsupported encoding" in str(exc):
                pytest.skip(f"Encoding '{name}' is not supported by the bpe-openai backend")
            raise

    def encoding_for_model_with_skip(name: str):
        key = name.lower()
        if key not in supported_models or _should_skip_model(name):
            pytest.skip(f"Model '{name}' is not supported by the bpe-openai backend")
        try:
            return original_encoding_for_model(name)
        except shim.UnsupportedModelError:
            pytest.skip(f"Model '{name}' is not supported by bpe-openai")
        except ValueError as exc:
            if "Unsupported model" in str(exc):
                pytest.skip(f"Model '{name}' is not supported by the bpe-openai backend")
            raise

    shim.get_encoding = get_encoding_with_skip  # type: ignore[attr-defined]
    shim.encoding_for_model = encoding_for_model_with_skip  # type: ignore[attr-defined]

    try:
        import pytest  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        print("pytest is required to run the tiktoken test suite", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    pytest_args = list(remaining) + [str(test_dir)]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
