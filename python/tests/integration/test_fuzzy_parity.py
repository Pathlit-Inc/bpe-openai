from __future__ import annotations

import pytest

import bpe_openai as candidate

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

SAMPLES = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.",
    "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque.",
    "On ne voit bien qu'avec le cœur. L'essentiel est invisible pour les yeux.",
    "迅速な茶色の狐が怠惰な犬を飛び越える。",
    "🌟🚀✨ Reaching for the stars requires more than luck—it takes preparation, grit, and teamwork!",
    "L'homme est libre au moment qu'il veut l'être.",
    "보라, 세계는 넓고 할 일은 많다.",
    "aaaaa" * 1000,
    "😀" * 512,
]


@pytest.mark.skipif(tiktoken is None, reason="tiktoken not available for parity check")
@pytest.mark.parametrize("sample", SAMPLES, ids=[f"sample_{i}" for i in range(len(SAMPLES))])
def test_fuzzy_parity_matches_tiktoken(sample: str) -> None:
    try:
        tk_encoding = tiktoken.encoding_for_model("gpt-4o")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiktoken unavailable: {exc}")

    candidate_encoding = candidate.encoding_for_model("gpt-4o")

    tk_tokens = tk_encoding.encode(sample)
    candidate_tokens = candidate_encoding.encode(sample)
    assert candidate_tokens == tk_tokens, f"token mismatch for text: {sample!r}"
    assert candidate_encoding.decode(candidate_tokens) == tk_encoding.decode(tk_tokens)
