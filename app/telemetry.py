"""Tracer setup.

Three exporters, one switch:

``memory``  keeps finished spans in a list. This is what the tests use, and what
            ``main.py`` uses to print the span tree, because it lets the process
            read its own telemetry back.
``console`` prints each span as JSON when it ends.
``otlp``    ships spans over OTLP/HTTP to whatever collector or backend you point
            it at.

The provider is returned rather than installed globally. Global tracer providers
are a well-known source of test pollution - the second call to
``set_tracer_provider`` in a process is silently ignored, so a test suite that
configures tracing twice quietly exports into the first run's exporter. Passing
the tracer explicitly costs one argument and removes the whole class of problem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

SERVICE_NAME = "travel-advisor"

# GenAI semantic-convention attribute names, kept in one place so the pipeline
# code does not scatter string literals.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Attributes with no standard name yet, namespaced so they are obviously ours.
APP_PATH = "advisor.path"
APP_COST_USD = "advisor.cost_usd"
APP_TOOL = "advisor.tool.name"
APP_TOOL_ARGS = "advisor.tool.arguments"
APP_CHUNKS = "advisor.retrieval.chunk_ids"
APP_TOP_SCORE = "advisor.retrieval.top_score"
APP_MEAN_SCORE = "advisor.retrieval.mean_score"
APP_CONTEXT_CHARS = "advisor.context.characters"
APP_ITERATION = "advisor.agent.iteration"


@dataclass
class Telemetry:
    """Everything a caller needs to emit and then read back spans."""

    provider: TracerProvider
    tracer: object
    exporter: Optional[InMemorySpanExporter] = None

    def finished_spans(self) -> List[object]:
        """Spans collected so far. Empty unless the memory exporter is in use."""

        self.provider.force_flush()
        if self.exporter is None:
            return []
        return list(self.exporter.get_finished_spans())

    def reset(self) -> None:
        if self.exporter is not None:
            self.exporter.clear()

    def shutdown(self) -> None:
        self.provider.shutdown()


def configure(exporter: str = "memory", service_name: str = SERVICE_NAME) -> Telemetry:
    """Build a tracer provider with the requested exporter."""

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.environ.get("APP_ENV", "local"),
        }
    )
    provider = TracerProvider(resource=resource)

    memory_exporter: Optional[InMemorySpanExporter] = None
    if exporter == "memory":
        memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    elif exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        provider.add_span_processor(SimpleSpanProcessor(_otlp_exporter()))
    elif exporter == "both":
        memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
        provider.add_span_processor(SimpleSpanProcessor(_otlp_exporter()))
    else:
        raise ValueError(
            "unknown exporter {!r}; expected memory, console, otlp or both".format(exporter)
        )

    return Telemetry(
        provider=provider,
        tracer=provider.get_tracer("travel-advisor"),
        exporter=memory_exporter,
    )


def _otlp_exporter():
    """Build the OTLP/HTTP exporter, with a useful message if it is missing.

    Endpoint and headers come from the standard environment variables, so this
    works against any OTLP-speaking backend without code changes:

        OTEL_EXPORTER_OTLP_ENDPOINT=https://collector.example.com
        OTEL_EXPORTER_OTLP_HEADERS=authorization=Api-Token%20<token>

    Credentials belong in those variables, never in this repository.
    """

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "OTLP export needs the exporter package: "
            "pip install opentelemetry-exporter-otlp-proto-http"
        ) from exc
    return OTLPSpanExporter()
