# bpe-openai

This directory contains the Python package that exposes the Rust `bpe-openai`
tokenizer via PyO3 bindings. The package is a thin wrapper over the Rust crate;
the compiled extension must be built before running tests or using the API.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt  # optional, see below
maturin develop --release
pytest
```

Running `maturin develop --release` builds the Rust extension in-place so the
package can be imported locally.

## Packaging

```bash
./python/scripts/build_wheels.sh
```

Requirements:

- Docker daemon access (the script exits if it cannot talk to `/var/run/docker.sock`).
- Sufficient permissions to pull `quay.io/pypa/manylinux2014_x86_64` (override with
  `MANYLINUX_IMAGE` if needed).
- Optional: set `MANYLINUX_PYTHON` to pick a different interpreter inside the
  container (defaults to `/opt/python/cp39-cp39/bin/python3.9`).

The container installs `maturin` on the fly and runs `maturin build --release`,
producing wheels under `python/target/wheels/` that meet manylinux requirements
and therefore run on both old and new glibc releases.

Additional scripts are mirrored at the repo root (`scripts/compare_with_tiktoken.py`,
`scripts/benchmark_scaling.py`) for convenience when working outside the
`python/` directory.

## Parity Comparison

To compare this package’s behaviour with upstream `tiktoken`, install
`tiktoken` in a virtual environment alongside the wheel and run:

```bash
./python/scripts/compare_with_tiktoken.py
```

The command prints per-model encode/decode parity summaries and highlights any
differences. Pass `--json` to capture machine-readable output.

To inspect performance scaling, run:

```bash
./scripts/benchmark_scaling.py --output python/target/benchmark_scaling.png
```

The script measures encode times for increasing input sizes and plots the
results if matplotlib is installed (mirrored at the repo root for convenience).

To inspect performance scaling, run:

```bash
./python/scripts/benchmark_scaling.py --output python/target/benchmark_scaling.png
```

This measures encoding time for increasing input sizes and, when matplotlib is
available, plots the results so you can compare the linear scaling of
`bpe_openai` with `tiktoken`.

## tiktoken Compatibility Snapshot

| Feature / API                              | Support Status | Notes |
|--------------------------------------------|----------------|-------|
| `encoding_for_model`                       | ✅              | Returns `Encoding` for all supported model aliases |
| `get_encoding`                             | ✅              | Supports `cl100k_base`, `o200k_base`, `voyage3_base` |
| `Encoding.encode`                          | ✅              | Calls the Rust `bpe-openai` tokenizer for exact parity |
| `Encoding.decode`                          | ✅              | Mirrors tiktoken behaviour (UTF-8 validation included) |
| `Encoding.encode_batch` / `decode_batch`   | ✅              | Batch helpers mirror `tiktoken` signatures |
| Unsupported model errors                   | ✅              | Raises `UnsupportedModelError` with `supported_models` list |
| Custom special tokens                      | ❌ Not yet      | Registering extra special tokens is not supported |
| Chunk limit enforcement                    | ✅              | Uses tokenizer counts and raises `TokenLimitError` |
| Metrics hook (`set_metrics_hook`)          | ✅              | Emits model, latency, token counts |
| Rust-backed performance parity             | ✅              | Wheels/binaries use the upstream Rust implementation |
| Quickstart smoke test (drop-in replacement)| ✅              | Existing scripts run unmodified in tests |

Legend: ✅ fully supported • ⚠️ requires follow-up (documented limitation) • ❌ not yet implemented |
