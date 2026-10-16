"""Embeddings and vector search, as their own spans.

Retrieval is where most bad answers actually come from, and it is invisible if
you only time the outer HTTP request. Giving the search its own span, carrying
the chunk ids it returned and the scores it assigned them, is what turns "the
answer was wrong" into "the retriever returned nothing above 0.2".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from opentelemetry.trace import Status, StatusCode

from .llm import embed, hashed_embedding
from .telemetry import (
    APP_CHUNKS,
    APP_CONTEXT_CHARS,
    APP_CONTEXT_CHUNKS,
    APP_CORPUS_SIZE,
    APP_MEAN_SCORE,
    APP_TOP_K,
    APP_TOP_SCORE,
)

CORPUS = {
    "kyoto-01": "Kyoto is busiest during cherry blossom season in early April and during autumn foliage in November.",
    "kyoto-02": "Most Kyoto temples open at 9am; Fushimi Inari is open all night and is quietest before sunrise.",
    "lisbon-01": "Lisbon is mild year round; July and August are hot and crowded, and May and October are the easiest months.",
    "lisbon-02": "Lisbon trams accept a rechargeable Viva Viagem card, which is cheaper than buying single tickets.",
    "reykjavik-01": "Reykjavik has long daylight in June and almost none in December; the aurora needs dark, clear nights.",
    "reykjavik-02": "Renting a car in Iceland in winter requires winter tyres, and many highland roads are closed until June.",
    "visa-01": "Most EU-bound travellers on a short stay need no visa, but passports must be valid for three months after departure.",
}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    score: float


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """An in-memory store over the corpus above."""

    def __init__(self) -> None:
        self.ids: List[str] = sorted(CORPUS)
        self.vectors = {key: hashed_embedding(CORPUS[key]) for key in self.ids}

    def search(self, query_vector: Sequence[float], top_k: int = 3) -> List[Chunk]:
        scored: List[Tuple[float, str]] = []
        for chunk_id in self.ids:
            scored.append((cosine(query_vector, self.vectors[chunk_id]), chunk_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            Chunk(chunk_id=chunk_id, text=CORPUS[chunk_id], score=score)
            for score, chunk_id in scored[:top_k]
        ]


def retrieve(tracer, store: VectorStore, question: str, top_k: int = 3) -> List[Chunk]:
    """Embed the question and search, as a parent span over the embedding span."""

    with tracer.start_as_current_span("retrieve") as span:
        span.set_attribute(APP_TOP_K, top_k)
        span.set_attribute(APP_CORPUS_SIZE, len(store.ids))

        query_vector = embed(tracer, question)

        with tracer.start_as_current_span("vector_search") as search_span:
            time.sleep(0.003)
            chunks = store.search(query_vector, top_k=top_k)
            scores = [chunk.score for chunk in chunks]
            search_span.set_attribute(APP_CHUNKS, [chunk.chunk_id for chunk in chunks])
            search_span.set_attribute(APP_TOP_SCORE, max(scores) if scores else 0.0)
            search_span.set_attribute(
                APP_MEAN_SCORE, sum(scores) / len(scores) if scores else 0.0
            )
            search_span.set_status(Status(StatusCode.OK))

        span.set_attribute(APP_TOP_SCORE, max(scores) if scores else 0.0)
        span.set_attribute(APP_MEAN_SCORE, sum(scores) / len(scores) if scores else 0.0)
        return chunks


def build_context(tracer, chunks: Sequence[Chunk]) -> str:
    """Assemble the prompt context, as its own span.

    Worth separating from retrieval: truncation and over-stuffing happen here,
    not in the search, and they look identical from the outside.
    """

    with tracer.start_as_current_span("assemble_context") as span:
        context = "\n".join("[{}] {}".format(c.chunk_id, c.text) for c in chunks)
        span.set_attribute(APP_CONTEXT_CHUNKS, len(chunks))
        span.set_attribute(APP_CONTEXT_CHARS, len(context))
        return context
