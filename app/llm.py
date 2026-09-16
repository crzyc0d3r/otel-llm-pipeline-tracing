"""Model calls, wrapped in GenAI-convention spans.

The "model" is deterministic and local. That is on purpose: this repository is
about the instrumentation, and a real provider would make every trace different,
make the tests flaky, and require a key to run. Latency is simulated so span
durations are meaningful, and token counts and costs are computed the same way a
real integration would report them.

Each call emits one span named ``<operation> <model>``, which is the GenAI
semantic-convention shape, carrying ``gen_ai.*`` attributes plus a cost attribute
of our own.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, Optional, Sequence

from opentelemetry.trace import Status, StatusCode

from .telemetry import (
    APP_COST_USD,
    GEN_AI_INPUT_TOKENS,
    GEN_AI_OPERATION,
    GEN_AI_OUTPUT_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
)

SYSTEM = "advisor"
CHAT_MODEL = "advisor-chat-mini"
EMBED_MODEL = "advisor-embed-small"

# USD per token, input and output. Same shape as a real price list.
PRICING: Dict[str, Sequence[float]] = {
    CHAT_MODEL: (0.15e-6, 0.60e-6),
    EMBED_MODEL: (0.02e-6, 0.0),
}


def count_tokens(text: str) -> int:
    """A four-characters-per-token approximation. Good enough for cost sums."""

    return max(1, len(text) // 4)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = PRICING.get(model, (0.0, 0.0))
    return input_tokens * input_rate + output_tokens * output_rate


def chat(
    tracer,
    prompt: str,
    model: str = CHAT_MODEL,
    latency: float = 0.012,
    attributes: Optional[Dict[str, object]] = None,
    fail: bool = False,
) -> str:
    """One chat completion, as one span."""

    with tracer.start_as_current_span("chat {}".format(model)) as span:
        span.set_attribute(GEN_AI_SYSTEM, SYSTEM)
        span.set_attribute(GEN_AI_OPERATION, "chat")
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)

        input_tokens = count_tokens(prompt)
        span.set_attribute(GEN_AI_INPUT_TOKENS, input_tokens)

        time.sleep(latency)

        if fail:
            error = RuntimeError("upstream model returned 503")
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise error

        answer = _compose(prompt)
        output_tokens = count_tokens(answer)
        span.set_attribute(GEN_AI_OUTPUT_TOKENS, output_tokens)
        span.set_attribute(APP_COST_USD, cost_usd(model, input_tokens, output_tokens))
        span.set_status(Status(StatusCode.OK))
        return answer


def embed(
    tracer,
    text: str,
    model: str = EMBED_MODEL,
    latency: float = 0.004,
) -> list:
    """One embedding call, as one span."""

    with tracer.start_as_current_span("embeddings {}".format(model)) as span:
        span.set_attribute(GEN_AI_SYSTEM, SYSTEM)
        span.set_attribute(GEN_AI_OPERATION, "embeddings")
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
        input_tokens = count_tokens(text)
        span.set_attribute(GEN_AI_INPUT_TOKENS, input_tokens)

        time.sleep(latency)

        vector = hashed_embedding(text)
        span.set_attribute(APP_COST_USD, cost_usd(model, input_tokens, 0))
        span.set_status(Status(StatusCode.OK))
        return vector


EMBED_DIM = 64


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> list:
    """Deterministic hashed bag-of-words vector, L2-normalised."""

    vector = [0.0] * dim
    for word in _words(text):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = digest[0] % dim
        sign = 1.0 if digest[1] % 2 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _words(text: str) -> list:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [word for word in cleaned.split() if word]


def _compose(prompt: str) -> str:
    """A stable, prompt-dependent answer so traces are reproducible."""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    topic = _words(prompt)[:6]
    return "Based on the information available, here is advice on {} [{}].".format(
        " ".join(topic) or "your trip", digest
    )
