#!/usr/bin/env python3
"""Compare bpe_openai outputs against tiktoken for quick parity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

try:
    import tiktoken  # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "tiktoken is required for this comparison script.\n"
        "Install it with `pip install tiktoken` and re-run.\n"
    )
    raise SystemExit(1) from exc

import bpe_openai

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "performance" / "fixtures"


PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    (FIXTURES / "long_form_prompt.txt").read_text(encoding="utf-8")
    if (FIXTURES / "long_form_prompt.txt").exists()
    else "Long form prompt unavailable",
]

MODELS = ["gpt-4o", "cl100k_base"]


def compare_model(model: str) -> dict[str, object]:
    """Compare encode/decode behaviour for a specific model."""
    report: dict[str, object] = {"model": model, "matches": True, "cases": []}
    try:
        tk_encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        tk_encoding = tiktoken.get_encoding(model)
    except Exception as exc:
        report["matches"] = False
        report["cases"].append({
            "prompt": "n/a",
            "error": f"Failed to load tiktoken encoding: {exc}",
        })
        return report
    bo_encoding = bpe_openai.encoding_for_model(model)

    # encode / decode parity
    for prompt in PROMPTS:
        tk_tokens = tk_encoding.encode(prompt)
        bo_tokens = bo_encoding.encode(prompt)
        comparison = {
            "prompt": prompt[:40] + ("…" if len(prompt) > 40 else ""),
            "encode_match": tk_tokens == bo_tokens,
            "token_count": len(tk_tokens),
        }
        if tk_tokens != bo_tokens:
            comparison["token_delta"] = len(bo_tokens) - len(tk_tokens)
        tk_text = tk_encoding.decode(tk_tokens)
        bo_text = bo_encoding.decode(bo_tokens)
        comparison["decode_match"] = tk_text == bo_text
        if tk_text != bo_text:
            comparison["decode_mismatch"] = {
                "tiktoken": tk_text,
                "bpe_openai": bo_text,
            }
        if not (comparison["encode_match"] and comparison["decode_match"]):
            report["matches"] = False
        report["cases"].append(comparison)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of human-readable summary",
    )
    args = parser.parse_args()

    results = [compare_model(model) for model in MODELS]

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for result in results:
            status = "✅" if result["matches"] else "❌"
            print(f"{status} {result['model']}")
            for case in result["cases"]:
                prompt = case["prompt"]
                encode_match = "match" if case["encode_match"] else "DIFF"
                decode_match = "match" if case["decode_match"] else "DIFF"
                print(f"  • {prompt}: encode={encode_match}, decode={decode_match}")
                if not case["encode_match"] and "token_delta" in case:
                    print(f"    token count delta: {case['token_delta']}")
                if not case["decode_match"] and "decode_mismatch" in case:
                    print("    decode differences:")
                    diff = case["decode_mismatch"]
                    print(f"      tiktoken   : {diff['tiktoken'][:80]}")
                    print(f"      bpe_openai : {diff['bpe_openai'][:80]}")
            print()


if __name__ == "__main__":  # pragma: no cover
    main()
