"""The analysis layer: tree reconstruction, totals and findings."""

from types import SimpleNamespace

import pytest
from opentelemetry.trace import StatusCode

from app import paths
from app.analysis import (
    find_problems,
    format_summary,
    group_by_trace,
    render_tree,
    summarise,
)
from app.retrieval import VectorStore
from app.telemetry import configure

QUESTION = "What should I pack for Kyoto, and what do hotels cost there?"


def test_each_request_is_one_trace(telemetry, tracer):
    paths.basic(tracer, QUESTION)
    paths.basic(tracer, "Something else entirely?")
    grouped = group_by_trace(telemetry.finished_spans())
    assert len(grouped) == 2
    assert all(len(spans) == 2 for spans in grouped.values())


def test_tree_indents_children_under_parents(telemetry, tracer):
    paths.rag(tracer, QUESTION, store=VectorStore())
    tree = render_tree(telemetry.finished_spans())
    lines = tree.splitlines()

    assert lines[0].startswith("advisor.rag")
    retrieve = next(line for line in lines if line.strip().startswith("retrieve"))
    search = next(line for line in lines if line.strip().startswith("vector_search"))
    assert _indent(search) > _indent(retrieve) > 0


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def test_summary_totals_the_model_usage(telemetry, tracer):
    paths.rag(tracer, QUESTION, store=VectorStore())
    summary = summarise(telemetry.finished_spans())

    assert summary.root_name == "advisor.rag"
    assert summary.span_count == 6
    assert summary.model_calls == 2  # one embedding, one chat
    assert summary.input_tokens > 0
    assert summary.output_tokens > 0
    assert summary.cost_usd > 0.0
    assert summary.wall_ms > 0.0


def test_repeated_tool_calls_are_flagged(telemetry, tracer):
    paths.agentic(tracer, QUESTION, max_iterations=5, remember_calls=False)
    findings = find_problems(telemetry.finished_spans())
    assert any("repeated tool call" in finding for finding in findings)


def test_a_healthy_agentic_run_is_not_flagged_for_repeats(telemetry, tracer):
    paths.agentic(tracer, QUESTION, remember_calls=True)
    findings = find_problems(telemetry.finished_spans())
    assert not any("repeated tool call" in finding for finding in findings)


def _top_score(spans):
    search = next(s for s in spans if s.name == "vector_search")
    return search.attributes["advisor.retrieval.top_score"]


def test_off_corpus_questions_retrieve_worse_and_still_succeed(telemetry, tracer):
    """The failure mode tracing exists for: a wrong answer with no error anywhere."""

    store = VectorStore()
    paths.rag(tracer, "When is Kyoto busiest with cherry blossom crowds?", store=store)
    on_corpus_spans = telemetry.finished_spans()
    on_corpus = _top_score(on_corpus_spans)
    telemetry.reset()

    paths.rag(tracer, "What is the tipping etiquette in Ulaanbaatar?", store=store)
    off_corpus_spans = telemetry.finished_spans()
    off_corpus = _top_score(off_corpus_spans)

    assert off_corpus < on_corpus
    # Neither request failed. Only the retrieval span distinguishes them.
    assert summarise(on_corpus_spans).errors == 0
    assert summarise(off_corpus_spans).errors == 0


def test_low_relevance_crosses_into_a_finding():
    """The threshold itself, checked without depending on hash behaviour."""

    good = _fake("vector_search", {"advisor.retrieval.mean_score": 0.61})
    bad = _fake("vector_search", {"advisor.retrieval.mean_score": 0.04})
    assert not any("weak retrieval" in f for f in find_problems([good]))
    assert any("weak retrieval" in f for f in find_problems([bad]))


def _fake(name, attributes=None, tool=None, args=None):
    """A minimal stand-in for a ReadableSpan, for testing the checks directly."""

    payload = dict(attributes or {})
    if tool:
        payload["advisor.tool.name"] = tool
        payload["advisor.tool.arguments"] = args or ""
    return SimpleNamespace(
        name=name,
        attributes=payload,
        parent=None,
        context=SimpleNamespace(span_id=1, trace_id=7),
        start_time=0,
        end_time=1_000_000,
        status=SimpleNamespace(status_code=StatusCode.UNSET, description=""),
    )


def test_identical_tool_calls_are_the_repeat_signal():
    spans = [_fake("tool lookup_weather", tool="lookup_weather", args="city=kyoto")] * 3
    findings = find_problems(spans)
    assert any("ran 3 times" in finding for finding in findings)


def test_same_tool_different_arguments_is_not_a_repeat():
    spans = [
        _fake("tool lookup_weather", tool="lookup_weather", args="city=kyoto"),
        _fake("tool lookup_weather", tool="lookup_weather", args="city=lisbon"),
    ]
    assert not any("repeated tool call" in finding for finding in find_problems(spans))


def test_failed_spans_are_reported(telemetry, tracer):
    from app.tools import call_tool

    with pytest.raises(KeyError):
        with tracer.start_as_current_span("advisor.agentic"):
            call_tool(tracer, "lookup_weather", "atlantis")

    summary = summarise(telemetry.finished_spans())
    assert summary.errors >= 1
    assert any("failed span" in finding for finding in summary.findings)


def test_empty_input_does_not_crash():
    assert render_tree([]) == "(no spans)"
    summary = summarise([])
    assert summary.span_count == 0
    assert "findings: none" in format_summary(summary)


def test_format_summary_lists_findings(telemetry, tracer):
    paths.agentic(tracer, QUESTION, max_iterations=4, remember_calls=False)
    text = format_summary(summarise(telemetry.finished_spans()))
    assert "findings:" in text
    assert "repeated tool call" in text


def test_console_exporter_configures_without_a_memory_collector():
    instance = configure("console")
    try:
        assert instance.exporter is None
        assert instance.finished_spans() == []
    finally:
        instance.shutdown()


def test_unknown_exporter_is_rejected():
    with pytest.raises(ValueError):
        configure("carrier-pigeon")
