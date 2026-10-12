"""Agent 3 - CHUNK_VALIDATOR_AGENT.

The quality gate between chunking and the vector/vectorless store: no chunk
reaches storage without passing through here. Scores come from the brain
(NIM LLM judge, or heuristic fallback); the gate policy itself is enforced
here - a chunk fails below 0.6 on any axis, a batch fails above 15% failures.
"""
from __future__ import annotations

from ..contracts import Chunk, ChunkScore, ChunkValidationResult


def validate_chunks(chunks: list[Chunk], brain) -> ChunkValidationResult:
    if not chunks:
        return ChunkValidationResult(
            stage="chunk_validation", status="fail", confidence=0.0,
            batch_pass_rate=0.0, chunk_scores=[], retry_feedback="no chunks to validate",
            reason="empty chunk batch",
        )
    result = brain.validate_chunks(chunks)

    # Gate policy from the prompt pack.
    scored = {s.chunk_id: s for s in result.chunk_scores}
    for c in chunks:
        if c.chunk_id not in scored:  # LLM missed a chunk: treat as failing, explicit fix.
            scored[c.chunk_id] = ChunkScore(
                chunk_id=c.chunk_id, coherence=0.0, boundary_quality=0.0,
                coverage=0.0, passed=False,
                suggested_fix=f"judge returned no score for {c.chunk_id}; re-score or re-chunk it",
            )
    result.chunk_scores = [scored[c.chunk_id] for c in chunks]

    n = len(chunks)
    n_fail = sum(1 for s in result.chunk_scores if not s.passed)
    result.batch_pass_rate = round((n - n_fail) / n, 3)
    if n_fail / n > 0.15:
        result.status = "fail"
    if result.status == "fail" and not result.retry_feedback:
        result.retry_feedback = (
            f"{n_fail}/{n} chunks failed the quality gate; fix them per the suggested_fix entries"
        )
    result.reason = f"{n - n_fail}/{n} chunks passed (threshold: <=15% failures)"
    return result
