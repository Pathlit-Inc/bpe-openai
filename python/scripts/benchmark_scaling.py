#!/usr/bin/env python3
"""Compare encoding scaling between tiktoken and bpe_openai and plot timings."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, List

try:
    import tiktoken  # type: ignore
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "tiktoken is required for this benchmark. Install it with `pip install tiktoken`.\n"
    )
    raise SystemExit(1) from exc

import bpe_openai

DEFAULT_LENGTHS = [50, 200, 1_000, 5_000, 10_000, 25_000, 50_000]
REPEAT = 5


def measure(encoding, text: str) -> float:
    start = time.perf_counter()
    encoding.encode(text)
    return (time.perf_counter() - start) * 1_000


def benchmark(lengths: Iterable[int]) -> List[dict[str, float]]:
    results: List[dict[str, float]] = []

    tk_encoding = tiktoken.encoding_for_model("gpt-4o")
    bo_encoding = bpe_openai.encoding_for_model("gpt-4o")

    for length in lengths:
        text = "A" * length

        tk_timings = [measure(tk_encoding, text) for _ in range(REPEAT)]
        bo_timings = [measure(bo_encoding, text) for _ in range(REPEAT)]

        results.append(
            {
                "length": length,
                "tiktoken_ms": statistics.mean(tk_timings),
                "bpe_openai_ms": statistics.mean(bo_timings),
            }
        )

    return results


def maybe_plot(results: List[dict[str, float]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        sys.stderr.write("matplotlib not installed; skipping plot.\n")
        return

    lengths = [row["length"] for row in results]
    tk_times = [row["tiktoken_ms"] for row in results]
    bo_times = [row["bpe_openai_ms"] for row in results]

    plt.figure(figsize=(8, 5))
    plt.plot(lengths, tk_times, marker="o", label="tiktoken")
    plt.plot(lengths, bo_times, marker="o", label="bpe_openai")
    plt.xlabel("Input length (characters)")
    plt.ylabel("Encode time (ms)")
    plt.title("Encoding time vs. input length")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        nargs="*",
        type=int,
        default=DEFAULT_LENGTHS,
        help="Custom lengths to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("python/target/benchmark_scaling.png"),
        help="Path to write the plot (default: %(default)s)",
    )
    args = parser.parse_args()

    results = benchmark(args.lengths)

    print("length\ttiktoken_ms\tbpe_openai_ms")
    for row in results:
        print(f"{row['length']}\t{row['tiktoken_ms']:.3f}\t{row['bpe_openai_ms']:.3f}")

    maybe_plot(results, args.output)


if __name__ == "__main__":
    main()
