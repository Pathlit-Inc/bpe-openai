from __future__ import annotations

from typing import Any, Dict, List

import bpe_openai as candidate


def test_metrics_hook_receives_token_and_timing_data() -> None:
    payloads: List[Dict[str, Any]] = []

    encoding = candidate.encoding_for_model("gpt-4o")
    encoding.set_metrics_hook(payloads.append)  # type: ignore[attr-defined]

    encoding.encode("monitor this")

    assert payloads, "metrics hook must be invoked"
    entry = payloads[-1]
    assert entry["model"] == "gpt-4o"
    assert entry["total_tokens"] > 0
    assert entry["elapsed_ms"] >= 0
