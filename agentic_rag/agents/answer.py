"""Agent 6 - ANSWER_AGENT.

Only ever called with context that has already passed
RETRIEVAL_VALIDATOR_AGENT. Answers from the validated context only, with
per-claim citations and explicit unanswered parts - never fabrication.
"""
from __future__ import annotations

from ..contracts import AnswerResult, RetrievedChunk


def generate_answer(query: str, retrieved: list[RetrievedChunk], brain) -> AnswerResult:
    result = brain.generate_answer(query, retrieved)
    result.reason = (
        f"{len(result.citations)} citation(s) over {len(retrieved)} validated chunk(s)"
        + (f" | {result.reason}" if result.reason else "")
    )
    if not result.answer:
        result.status = "fail"
    return result
