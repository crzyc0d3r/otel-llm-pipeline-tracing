"""Read the collected spans back and say what they show.

A real deployment ships spans to a backend and you read them there. This module
exists so the repository can demonstrate the payoff without one: it consumes the
same ReadableSpan objects an exporter would, reconstructs the trace tree, and
reports the things a human would go looking for.

The findings are deliberately mundane. Nothing here needs a backend, a query
language, or a dashboard - it needs the spans to exist and to carry the right
attributes, which is the actual work.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from opentelemetry.trace import StatusCode

from .telemetry import (
    APP_COST_USD,
    APP_MEAN_SCORE,
    APP_TOOL,
    APP_TOOL_ARGS,
    GEN_AI_INPUT_TOKENS,
    GEN_AI_OPERATION,
    GEN_AI_OUTPUT_TOKENS,
)

NANOS_PER_MS = 1_000_000.0
LOW_RELEVANCE = 0.25


def span_id(span) -> Optional[int]:
    context = getattr(span, "context", None)
    if context is None and hasattr(span, "get_span_context"):
        context = span.get_span_context()
    return context.span_id if context is not None else None


def trace_id(span) -> Optional[int]:
    context = getattr(span, "context", None)
    if context is None and hasattr(span, "get_span_context"):
        context = span.get_span_context()
    return context.trace_id if context is not None else None


def parent_id(span) -> Optional[int]:
    parent = getattr(span, "parent", None)
    return parent.span_id if parent is not None else None


def duration_ms(span) -> float:
    start = getattr(span, "start_time", None)
    end = getattr(span, "end_time", None)
    if start is None or end is None:
        return 0.0
    return (end - start) / NANOS_PER_MS


def attribute(span, key, default=None):
    attributes = getattr(span, "attributes", None) or {}
    return attributes.get(key, default)


def is_error(span) -> bool:
    status = getattr(span, "status", None)
    return status is not None and status.status_code is StatusCode.ERROR


@dataclass
class Summary:
    """Per-trace totals."""

    trace_id: Optional[int]
    root_name: str
    span_count: int
    wall_ms: float
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    errors: int
    findings: List[str] = field(default_factory=list)


def group_by_trace(spans: Sequence[object]) -> Dict[Optional[int], List[object]]:
    grouped: Dict[Optional[int], List[object]] = defaultdict(list)
    for span in spans:
        grouped[trace_id(span)].append(span)
    return dict(grouped)


def _children(spans: Sequence[object]) -> Dict[Optional[int], List[object]]:
    children: Dict[Optional[int], List[object]] = defaultdict(list)
    for span in spans:
        children[parent_id(span)].append(span)
    for bucket in children.values():
        bucket.sort(key=lambda s: getattr(s, "start_time", 0) or 0)
    return children


def render_tree(spans: Sequence[object]) -> str:
    """One trace as an indented tree with durations."""

    if not spans:
        return "(no spans)"
    children = _children(spans)
    known = {span_id(span) for span in spans}
    roots = [span for span in spans if parent_id(span) not in known]
    roots.sort(key=lambda s: getattr(s, "start_time", 0) or 0)

    lines: List[str] = []

    def walk(span, depth: int) -> None:
        marker = "ERROR " if is_error(span) else ""
        lines.append(
            "{}{}{:<34} {:>8.1f} ms{}".format(
                "  " * depth, marker, span.name, duration_ms(span), _detail(span)
            )
        )
        for child in children.get(span_id(span), []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return "\n".join(lines)


def _detail(span) -> str:
    bits = []
    tool = attribute(span, APP_TOOL)
    if tool:
        bits.append(str(attribute(span, APP_TOOL_ARGS, "")))
    operation = attribute(span, GEN_AI_OPERATION)
    if operation:
        input_tokens = attribute(span, GEN_AI_INPUT_TOKENS, 0) or 0
        output_tokens = attribute(span, GEN_AI_OUTPUT_TOKENS, 0) or 0
        bits.append("{}+{} tok".format(input_tokens, output_tokens))
    mean_score = attribute(span, APP_MEAN_SCORE)
    if mean_score is not None:
        bits.append("mean score {:.3f}".format(float(mean_score)))
    return ("   " + ", ".join(bits)) if bits else ""


def summarise(spans: Sequence[object]) -> Summary:
    """Totals plus findings for one trace."""

    if not spans:
        return Summary(None, "(empty)", 0, 0.0, 0, 0, 0, 0, 0.0, 0)

    known = {span_id(span) for span in spans}
    roots = [span for span in spans if parent_id(span) not in known]
    root = min(roots, key=lambda s: getattr(s, "start_time", 0) or 0) if roots else spans[0]

    model_calls = [s for s in spans if attribute(s, GEN_AI_OPERATION)]
    tool_spans = [s for s in spans if attribute(s, APP_TOOL)]

    summary = Summary(
        trace_id=trace_id(root),
        root_name=root.name,
        span_count=len(spans),
        wall_ms=duration_ms(root),
        model_calls=len(model_calls),
        tool_calls=len(tool_spans),
        input_tokens=sum(int(attribute(s, GEN_AI_INPUT_TOKENS, 0) or 0) for s in model_calls),
        output_tokens=sum(int(attribute(s, GEN_AI_OUTPUT_TOKENS, 0) or 0) for s in model_calls),
        cost_usd=sum(float(attribute(s, APP_COST_USD, 0.0) or 0.0) for s in spans),
        errors=sum(1 for s in spans if is_error(s)),
    )
    summary.findings = find_problems(spans)
    return summary


def find_problems(spans: Sequence[object]) -> List[str]:
    """The checks worth running on every trace."""

    findings: List[str] = []

    repeats = Counter(
        (attribute(s, APP_TOOL), attribute(s, APP_TOOL_ARGS))
        for s in spans
        if attribute(s, APP_TOOL)
    )
    for (tool, args), count in sorted(repeats.items()):
        if count > 1:
            findings.append(
                "repeated tool call: {}({}) ran {} times with identical arguments".format(
                    tool, args, count
                )
            )

    for span in spans:
        mean_score = attribute(span, APP_MEAN_SCORE)
        if mean_score is not None and float(mean_score) < LOW_RELEVANCE:
            findings.append(
                "weak retrieval: '{}' returned a mean relevance of {:.3f}; "
                "the answer is probably ungrounded".format(span.name, float(mean_score))
            )

    for span in spans:
        if is_error(span):
            status = getattr(span, "status", None)
            description = getattr(status, "description", "") or "no description"
            findings.append("failed span: '{}' - {}".format(span.name, description))

    if spans:
        known = {span_id(s) for s in spans}
        leaves = [s for s in spans if not any(parent_id(o) == span_id(s) for o in spans)]
        if leaves:
            slowest = max(leaves, key=duration_ms)
            root_candidates = [s for s in spans if parent_id(s) not in known]
            total = duration_ms(root_candidates[0]) if root_candidates else 0.0
            share = (duration_ms(slowest) / total * 100.0) if total else 0.0
            if share >= 40.0:
                findings.append(
                    "one stage dominates: '{}' is {:.0f}% of the request".format(
                        slowest.name, share
                    )
                )

    return findings


def format_summary(summary: Summary) -> str:
    lines = [
        "{}  -  {} spans, {:.1f} ms wall".format(
            summary.root_name, summary.span_count, summary.wall_ms
        ),
        "  model calls {}   tool calls {}   tokens {}+{}   cost ${:.6f}   errors {}".format(
            summary.model_calls,
            summary.tool_calls,
            summary.input_tokens,
            summary.output_tokens,
            summary.cost_usd,
            summary.errors,
        ),
    ]
    if summary.findings:
        lines.append("  findings:")
        lines.extend("    - {}".format(finding) for finding in summary.findings)
    else:
        lines.append("  findings: none")
    return "\n".join(lines)
