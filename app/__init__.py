"""A travel advisor with three execution paths, instrumented end to end.

    app.telemetry  - tracer setup and exporter selection
    app.llm        - the model calls (deterministic stand-ins)
    app.retrieval  - embeddings and vector search
    app.tools      - the tools the agentic path can call
    app.paths      - basic / rag / agentic
    app.analysis   - read the collected spans back and say what went wrong
"""

__all__ = ["telemetry", "llm", "retrieval", "tools", "paths", "analysis"]
