"""Agent 5 - RETRIEVAL_VALIDATOR_AGENT.

The last gate before context reaches answer generation - nothing passes
through ungrounded. Scores come from the brain (NIM LLM judge or heuristic
fallback); the gate policy itself is enforced here:

  fail if sufficiency < 0.4 OR groundedness_risk > 0.65
       OR fewer than 35% of chunks are individually relevant.

Thresholds are calibrated for real-world documents (PDFs, reports) that
contain heading-only chunks which legitimately score low on relevance
without meaning the retrieval failed.
"""
from __future__ import annotations

from ..contracts import RetrievedChunk, RetrievalValidationResult


def validate_retrieval(query: str, retrieved: list[RetrievedChunk], brain) -> RetrievalValidationResult:
    result = brain.validate_retrieval(query, retrieved)

    fails = (
        result.sufficiency < 0.4
        or result.groundedness_risk > 0.65
        or (bool(retrieved) and result.relevant_share < 0.35)
    )
    result.status = "fail" if fails else "pass"

    if result.status == "fail" and not result.retry_feedback:
        result.retry_feedback = (
            f"sufficiency={result.sufficiency:.2f}, groundedness_risk={result.groundedness_risk:.2f}, "
            f"relevant_share={result.relevant_share:.2f}; adjust retrieval"
        )
    result.reason = (
        f"sufficiency={result.sufficiency:.3f}, groundedness_risk={result.groundedness_risk:.3f}, "
        f"relevance_avg={result.relevance_avg:.3f} | {result.reason}"
    )
    return result
