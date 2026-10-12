"""Agent 4 - RETRIEVAL_AGENT.

Hybrid retrieval: two complementary paths always run and are then merged.

- Path A (VECTOR): dense-embedding similarity over the validated chunk store
  (pgvector or in-memory). Best for open-ended semantic queries.
- Path B (VECTORLESS): reasoning over the document structure index (section
  paths) the way a person uses a table of contents - identify which sections
  plausibly contain the answer, then descend into those sections' chunks.
  Essential for multi-hop, comparative, and structural queries.

Every returned chunk is tagged with which path(s) surfaced it.
"""
from __future__ import annotations

import re

from ..contracts import RetrievedChunk, RetrievalResult
from ..store import VectorStore

_STRUCTURAL_RE = re.compile(r"section\s*(\d+)", re.IGNORECASE)


def retrieve(query: str, store: VectorStore, embedder, brain,
             top_k: int, retry_feedback: str | None = None) -> RetrievalResult:
    # Path A: vector similarity over validated chunks.
    query_embedding = embedder.embed_query(query)
    vector_hits = store.search(query_embedding, top_k)
    # A query that names a section is best served by the structure index; keep
    # a few vector hits for context but do not let them dilute the section pick.
    if _STRUCTURAL_RE.search(query):
        vector_hits = vector_hits[: max(4, top_k // 2)]

    # Path B: vectorless, structure-index based.
    sections = store.get_sections()
    chosen = brain.plan_vectorless(sections, query)
    # The document root prefixes every chunk, so descending into it would pull
    # the whole document and dilute the merged set. Structure reasoning only
    # makes sense at or below top-level sections.
    chosen = [p for p in chosen if " / " in p]
    if retry_feedback and "vectorless" in retry_feedback.lower():
        chosen = _broaden(chosen, sections)
    vectorless_hits: list = []
    for path in chosen:
        vectorless_hits.extend(store.chunks_in_section(path))

    # Merge: de-duplicate, keep highest combined relevance, tag source paths.
    merged: dict[str, RetrievedChunk] = {}
    for h in vector_hits:
        merged[h.chunk_id] = RetrievedChunk(
            chunk_id=h.chunk_id, text=h.text, section_path=h.section_path,
            source_path="vector", relevance_score=round(max(0.0, h.score), 3),
        )
    for h in vectorless_hits:
        if h.chunk_id in merged:
            merged[h.chunk_id].source_path = "both"
            merged[h.chunk_id].relevance_score = max(merged[h.chunk_id].relevance_score, 0.75)
        else:
            merged[h.chunk_id] = RetrievedChunk(
                chunk_id=h.chunk_id, text=h.text, section_path=h.section_path,
                source_path="vectorless", relevance_score=0.7,
            )

    combined = sorted(merged.values(), key=lambda c: c.relevance_score, reverse=True)
    combined = combined[: max(top_k * 2, 12)]

    n_v = sum(1 for c in combined if c.source_path == "vector")
    n_vl = sum(1 for c in combined if c.source_path == "vectorless")
    n_both = sum(1 for c in combined if c.source_path == "both")
    reason = (
        f"vector: {len(vector_hits)} hit(s), vectorless: {len(vectorless_hits)} hit(s) "
        f"from sections {chosen if chosen else '[]'}; merged -> {len(combined)} chunk(s) "
        f"(vector={n_v}, vectorless={n_vl}, both={n_both})"
    )
    return RetrievalResult(
        stage="retrieval",
        status="pass" if combined else "fail",
        confidence=0.9 if combined else 0.0,
        query=query,
        retrieved_chunks=combined,
        reason=reason,
    )


def _broaden(chosen: list[str], sections: list[str]) -> list[str]:
    """On validator feedback, also descend into parent sections of the picks.

    The document root is excluded: it prefixes every chunk, so pulling it in
    would dilute the retrieved set instead of enriching it.
    """
    out = list(chosen)
    for path in chosen:
        while " / " in path:
            path = path.rsplit(" / ", 1)[0]
            if " / " not in path:      # reached the document root - too broad
                break
            if path in sections and path not in out:
                out.append(path)
    return out[:6]
