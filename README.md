# bpe-openai

Rust-first reimplementation of OpenAI's byte pair encoding (BPE) tokenizers,
exposing a PyO3-powered Python extension. The project aims for feature parity
with `tiktoken` while keeping the tokenizer logic in Rust for speed and
portability.

## Repository layout
- `rust/` – core Rust crate that loads tokenizer specs and exposes fast BPE APIs.
- `python/` – Python package that builds the Rust crate into a CPython extension.
- `scripts/` – helper utilities for benchmarking and parity checks.
- `vendor/` – vendored tokenizer definitions sourced from upstream releases.

## Quick start
```bash
# Build and test the Rust crate
cd rust
cargo test

# Build the Python extension in-place and run the test suite
cd ../python
python -m venv .venv
source .venv/bin/activate
maturin develop --release
pytest
```

See `python/README.md` for packaging details (wheel builds, Docker-based manylinux
artifacts, and parity benchmarking guidance).

## Benchmark sample
```text
$ python scripts/benchmark_scaling.py
length  tiktoken_ms  bpe_openai_ms
50      5.819        0.059
200     0.087        0.025
1000    0.895        0.088
5000    18.794       0.332
10000   74.301       0.616
25000   321.519      0.936
50000   1241.213     2.110
```

## License
This project is licensed under the terms of the `LICENSE` file in this directory.
