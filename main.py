"""Run the travel advisor and show what its traces reveal.

    python main.py                       # all three paths, trees and findings
    python main.py --path rag
    python main.py --compare             # the agentic path with and without memory
    python main.py --exporter otlp       # ship to a collector instead of printing

Every path answers a question. The point is not the answers - it is that the
three look identical from outside the process and completely different inside.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app import paths
from app.analysis import format_summary, group_by_trace, render_tree, summarise
from app.retrieval import VectorStore
from app.telemetry import configure

DEFAULT_QUESTION = "What should I pack for Kyoto, and what do hotels cost there?"
OFF_CORPUS_QUESTION = "What is the tipping etiquette in Ulaanbaatar?"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", choices=("basic", "rag", "agentic", "all"), default="all")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--exporter",
        choices=("memory", "console", "otlp", "both"),
        default="memory",
        help="'memory' prints the tree here; 'otlp' ships to OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run the agentic path with and without call memory and diff the traces",
    )
    parser.add_argument(
        "--off-corpus",
        action="store_true",
        help="ask the RAG path something the corpus does not cover",
    )
    return parser


def _report(telemetry, label: str) -> None:
    spans = telemetry.finished_spans()
    if not spans:
        print("{}: no spans captured (exporter is not 'memory' or 'both')".format(label))
        return
    for trace_spans in group_by_trace(spans).values():
        print(render_tree(trace_spans))
        print()
        print(format_summary(summarise(trace_spans)))
        print()
    telemetry.reset()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    telemetry = configure(args.exporter)
    tracer = telemetry.tracer
    store = VectorStore()

    try:
        if args.compare:
            print("=" * 72)
            print("AGENTIC, remembering what it already called")
            print("=" * 72)
            paths.agentic(tracer, args.question, remember_calls=True)
            _report(telemetry, "agentic (fixed)")

            print("=" * 72)
            print("AGENTIC, not remembering - same answer, same success, more of everything")
            print("=" * 72)
            paths.agentic(tracer, args.question, remember_calls=False)
            _report(telemetry, "agentic (buggy)")
            return 0

        selected = ("basic", "rag", "agentic") if args.path == "all" else (args.path,)
        for name in selected:
            question = (
                OFF_CORPUS_QUESTION if (args.off_corpus and name == "rag") else args.question
            )
            print("=" * 72)
            print("{}  |  {}".format(name.upper(), question))
            print("=" * 72)
            if name == "basic":
                answer = paths.basic(tracer, question)
            elif name == "rag":
                answer = paths.rag(tracer, question, store=store)
            else:
                answer = paths.agentic(tracer, question)
            print("answer: {}".format(answer.text))
            print()
            _report(telemetry, name)
        return 0
    finally:
        telemetry.shutdown()


if __name__ == "__main__":
    sys.exit(main())
