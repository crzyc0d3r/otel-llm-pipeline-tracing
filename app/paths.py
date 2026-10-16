"""The three execution paths.

They answer the same kind of question and they look identical from outside the
process: one request in, one answer out, a couple of hundred milliseconds. The
whole argument for tracing is that "identical from outside" is exactly the
problem - the basic path makes one model call, the RAG path makes three calls
across two stages, and the agentic path makes a variable number that depends on
what the model decided to do.

Each path opens one root span named ``advisor.<path>``; everything else nests
under it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from opentelemetry.trace import Status, StatusCode

from .llm import chat
from .retrieval import VectorStore, build_context, retrieve
from .telemetry import (
    APP_CITIES,
    APP_DISTINCT_CALLS,
    APP_ITERATION,
    APP_ITERATIONS_USED,
    APP_MAX_ITERATIONS,
    APP_PATH,
)
from .tools import call_tool

CITIES = ("kyoto", "lisbon", "reykjavik")


@dataclass
class Answer:
    path: str
    question: str
    text: str
    tool_calls: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)


def basic(tracer, question: str) -> Answer:
    """One model call. The baseline everything else is compared against."""

    with tracer.start_as_current_span("advisor.basic") as span:
        span.set_attribute(APP_PATH, "basic")
        text = chat(tracer, question, attributes={APP_PATH: "basic"})
        span.set_status(Status(StatusCode.OK))
        return Answer(path="basic", question=question, text=text)


def rag(tracer, question: str, store: Optional[VectorStore] = None, top_k: int = 3) -> Answer:
    """Retrieve, assemble, generate."""

    store = store or VectorStore()
    with tracer.start_as_current_span("advisor.rag") as span:
        span.set_attribute(APP_PATH, "rag")
        chunks = retrieve(tracer, store, question, top_k=top_k)
        context = build_context(tracer, chunks)
        prompt = "Context:\n{}\n\nQuestion: {}".format(context, question)
        text = chat(tracer, prompt, attributes={APP_PATH: "rag"})
        span.set_status(Status(StatusCode.OK))
        return Answer(
            path="rag",
            question=question,
            text=text,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )


def agentic(
    tracer,
    question: str,
    max_iterations: int = 6,
    remember_calls: bool = True,
) -> Answer:
    """A tool-calling loop.

    ``remember_calls=False`` reproduces a specific, very common bug: the loop
    picks its next tool from the question alone and never checks what it has
    already done, so it re-runs the same tool with the same arguments until it
    hits the iteration cap. The answer it returns is still correct, the request
    still succeeds, and the only symptom is that it took four times as long and
    cost four times as much.

    That is the whole point of the exercise. Run both and diff the traces.
    """

    cities = [city for city in CITIES if city in question.lower()] or ["lisbon"]
    wanted = _plan(question)

    with tracer.start_as_current_span("advisor.agentic") as span:
        span.set_attribute(APP_PATH, "agentic")
        span.set_attribute(APP_MAX_ITERATIONS, max_iterations)
        span.set_attribute(APP_CITIES, list(cities))

        observations: List[str] = []
        performed: List[str] = []
        done: set = set()

        for iteration in range(1, max_iterations + 1):
            plan = _next_call(wanted, cities, done if remember_calls else set())
            if plan is None:
                break
            tool_name, city = plan

            with tracer.start_as_current_span("agent_iteration") as iteration_span:
                iteration_span.set_attribute(APP_ITERATION, iteration)
                result = call_tool(tracer, tool_name, city)
                observations.append("{} ({}): {}".format(tool_name, city, result))
                performed.append("{}:{}".format(tool_name, city))
                done.add((tool_name, city))

        prompt = "Question: {}\nObservations:\n{}".format(question, "\n".join(observations))
        text = chat(tracer, prompt, attributes={APP_PATH: "agentic"})

        span.set_attribute(APP_ITERATIONS_USED, len(performed))
        span.set_attribute(APP_DISTINCT_CALLS, len(done))
        span.set_status(Status(StatusCode.OK))
        return Answer(path="agentic", question=question, text=text, tool_calls=performed)


def _plan(question: str) -> List[str]:
    """Which tools the question implies. Keyword matching, deliberately simple."""

    lowered = question.lower()
    wanted: List[str] = []
    if any(word in lowered for word in ("weather", "rain", "warm", "cold", "pack")):
        wanted.append("lookup_weather")
    if any(word in lowered for word in ("flight", "fly", "getting there", "airfare")):
        wanted.append("lookup_flights")
    if any(word in lowered for word in ("hotel", "stay", "accommodation", "night")):
        wanted.append("lookup_hotels")
    return wanted or ["lookup_weather"]


def _next_call(wanted: Sequence[str], cities: Sequence[str], done: set):
    """Pick the next (tool, city) pair that has not been done yet.

    With an empty ``done`` set - which is what ``remember_calls=False`` passes -
    this always returns the first pair, forever.
    """

    for tool_name in wanted:
        for city in cities:
            if (tool_name, city) not in done:
                return tool_name, city
    return None
