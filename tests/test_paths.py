"""Each path emits the spans that make it debuggable."""

import pytest

from app import paths
from app.analysis import attribute, parent_id, span_id, summarise
from app.retrieval import VectorStore
from app.telemetry import (
    APP_PATH,
    APP_TOOL,
    GEN_AI_OPERATION,
    GEN_AI_REQUEST_MODEL,
)

QUESTION = "What should I pack for Kyoto, and what do hotels cost there?"


def names(spans):
    return [span.name for span in spans]


def test_basic_path_is_one_model_call(telemetry, tracer):
    answer = paths.basic(tracer, QUESTION)
    spans = telemetry.finished_spans()

    assert answer.path == "basic"
    assert len(spans) == 2
    assert "advisor.basic" in names(spans)
    assert any(attribute(s, GEN_AI_OPERATION) == "chat" for s in spans)


def test_rag_path_separates_every_stage(telemetry, tracer):
    """The whole argument: retrieval, assembly and generation are not one call."""

    answer = paths.rag(tracer, QUESTION, store=VectorStore())
    spans = telemetry.finished_spans()
    collected = names(spans)

    assert answer.path == "rag"
    assert len(spans) == 6
    for expected in ("advisor.rag", "retrieve", "vector_search", "assemble_context"):
        assert expected in collected
    assert any(attribute(s, GEN_AI_OPERATION) == "embeddings" for s in spans)
    assert any(attribute(s, GEN_AI_OPERATION) == "chat" for s in spans)


def test_rag_spans_nest_correctly(telemetry, tracer):
    paths.rag(tracer, QUESTION, store=VectorStore())
    spans = telemetry.finished_spans()
    by_name = {span.name: span for span in spans}

    root = by_name["advisor.rag"]
    retrieve = by_name["retrieve"]
    search = by_name["vector_search"]

    assert parent_id(root) is None
    assert parent_id(retrieve) == span_id(root)
    assert parent_id(search) == span_id(retrieve)


def test_retrieval_records_what_it_returned(telemetry, tracer):
    answer = paths.rag(tracer, QUESTION, store=VectorStore(), top_k=3)
    spans = telemetry.finished_spans()
    search = next(s for s in spans if s.name == "vector_search")

    chunk_ids = attribute(search, "advisor.retrieval.chunk_ids")
    assert len(chunk_ids) == 3
    assert list(chunk_ids) == answer.chunk_ids
    assert attribute(search, "advisor.retrieval.top_score") is not None


def test_model_spans_carry_the_genai_conventions(telemetry, tracer):
    paths.basic(tracer, QUESTION)
    spans = telemetry.finished_spans()
    chat = next(s for s in spans if attribute(s, GEN_AI_OPERATION) == "chat")

    assert attribute(chat, GEN_AI_REQUEST_MODEL)
    assert attribute(chat, "gen_ai.usage.input_tokens") > 0
    assert attribute(chat, "gen_ai.usage.output_tokens") > 0
    assert attribute(chat, APP_PATH) == "basic"


def test_agentic_path_emits_one_span_per_tool(telemetry, tracer):
    answer = paths.agentic(tracer, QUESTION, remember_calls=True)
    spans = telemetry.finished_spans()
    tool_spans = [s for s in spans if attribute(s, APP_TOOL)]

    assert answer.path == "agentic"
    assert len(tool_spans) == len(answer.tool_calls)
    assert {attribute(s, APP_TOOL) for s in tool_spans} == {
        "lookup_weather",
        "lookup_hotels",
    }


def test_the_question_decides_which_tools_run(telemetry, tracer):
    paths.agentic(tracer, "How are flights to Lisbon?", remember_calls=True)
    spans = telemetry.finished_spans()
    tools = {attribute(s, APP_TOOL) for s in spans if attribute(s, APP_TOOL)}
    assert tools == {"lookup_flights"}


def test_forgetting_costs_work_without_gathering_anything_new(telemetry, tracer):
    """Both runs succeed. Only the trace shows one of them doing three times the work."""

    fixed = paths.agentic(tracer, QUESTION, remember_calls=True)
    fixed_summary = summarise(telemetry.finished_spans())
    telemetry.reset()

    buggy = paths.agentic(tracer, QUESTION, max_iterations=6, remember_calls=False)
    buggy_summary = summarise(telemetry.finished_spans())

    # The buggy run calls a tool six times and learns one thing.
    assert len(set(buggy.tool_calls)) == 1
    assert buggy_summary.tool_calls == 6
    assert fixed_summary.tool_calls == 2

    # Neither run failed, so nothing upstream of the trace would flag it.
    assert buggy_summary.errors == 0
    assert fixed_summary.errors == 0

    # The cost is real and it is only visible here.
    assert buggy_summary.span_count > fixed_summary.span_count
    assert buggy_summary.input_tokens > fixed_summary.input_tokens


def test_tool_errors_are_recorded_not_swallowed(telemetry, tracer):
    from app.tools import call_tool

    with pytest.raises(KeyError):
        with tracer.start_as_current_span("probe"):
            call_tool(tracer, "lookup_weather", "atlantis")

    spans = telemetry.finished_spans()
    tool_span = next(s for s in spans if attribute(s, APP_TOOL) == "lookup_weather")
    assert tool_span.status.status_code.name == "ERROR"
