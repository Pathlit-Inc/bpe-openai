#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PY_DIR="$ROOT_DIR/python"
RUST_MANIFEST="$ROOT_DIR/rust/Cargo.toml"
WHEEL_DIR="$PY_DIR/target/wheels"

if [[ ! -f "$RUST_MANIFEST" ]]; then
  echo "Cannot locate Rust manifest at $RUST_MANIFEST" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build manylinux wheels. Install Docker and ensure you can access the daemon." >&2
  exit 1
fi

IMAGE="${MANYLINUX_IMAGE:-quay.io/pypa/manylinux2014_x86_64}"
PYTHON_BIN="${MANYLINUX_PYTHON:-/opt/python/cp39-cp39/bin/python3.9}"

mkdir -p "$WHEEL_DIR"

docker run --rm \
  -v "$ROOT_DIR":/io \
  -w /io/python \
  -e MANYLINUX_PYTHON_TARGET="$PYTHON_BIN" \
  "$IMAGE" \
  bash -lc 'set -euo pipefail; PY="${MANYLINUX_PYTHON_TARGET}"; if [ ! -x "$PY" ]; then echo "Expected interpreter $PY not found in container" >&2; exit 1; fi; "$PY" -m pip install --upgrade pip maturin; "$PY" -m maturin build --release --manifest-path /io/rust/Cargo.toml --out /io/python/target/wheels --interpreter "$PY"'

echo "Wheels written to $WHEEL_DIR"
