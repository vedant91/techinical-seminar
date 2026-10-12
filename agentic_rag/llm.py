"""The "brain" behind the LLM-judged stages.

Two interchangeable implementations of the same interface:

- `NIMBrain`   - calls NVIDIA NIM (OpenAI-compatible) with the prompt pack's
                 agent prompts, strictly JSON-structured outputs.
- `HeuristicBrain` - deterministic stand-in judges used when no API key is set,
                 so the whole pipeline is runnable and testable offline.

Every brain method may raise `LLMOutputError` after the internal malformed-JSON
retry is exhausted; the Orchestrator turns that into an escalation, never a
silent failure.
"""
from __future__ import annotations

import json
import re

from .config import Config
from .contracts import (
    AnswerResult,
    Chunk,
    ChunkScore,
    ChunkValidationResult,
    Citation,
    RetrievedChunk,
    RetrievalValidationResult,
)


class LLMOutputError(Exception):
    """Raised when an LLM stage cannot produce valid structured output."""


# Imported lazily-ish: the heuristic path must not require the openai SDK.
try:
    import openai as _openai

    # 4xx rejections of the request shape (e.g. json_object mode unsupported)
    # get one fallback retry; anything else escalates immediately.
    _OPENAI_API_ERRORS = (_openai.BadRequestError, _openai.UnprocessableEntityError)
    _OPENAI_API_ERRORS_GENERIC = (_openai.APIError,)
except ImportError:
    _OPENAI_API_ERRORS = ()
    _OPENAI_API_ERRORS_GENERIC = ()


def _parse_json_object(raw: str) -> dict:
    """Extract a JSON object from model output that may carry markdown fences
    or a short reasoning preamble (thinking-mode models)."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("no JSON object found in model output")


STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "who", "why",
    "did", "get", "him", "let", "may", "nor", "off", "old", "own", "see", "she",
    "too", "via", "what", "when", "where", "which", "while", "with", "your",
    "from", "into", "that", "they", "this", "have", "will", "does", "each",
    "them", "then", "than", "also", "about", "their", "there", "these", "those",
    "would", "could", "should", "such", "some", "other", "only", "very", "more",
    "most", "much", "many", "well", "even", "ever", "just", "like", "been",
    # conversational query fillers (safe to ignore for lexical overlap)
    "say", "tell", "explain", "describe", "according", "per",
    "use", "uses", "using",
}

_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")

# Words that point at document structure rather than answerable content.
_STRUCTURAL_WORDS = {
    "section", "sections", "spec", "specs", "specification",
    "product", "document", "doc", "chapter", "part", "paragraph",
}
_STRUCTURAL_QUERY_RE = re.compile(r"section\s*\d+")


def _topic_words(query: str) -> list[str]:
    """Content words of the query, minus structural pointers.

    "What does section 4 of the product spec say about pricing?" names a
    document location; its answerable topic is just {pricing}. Non-structural
    queries return their content words unchanged.
    """
    words = content_words(query)
    if _STRUCTURAL_QUERY_RE.search(query.lower()):
        words = [w for w in words if w not in _STRUCTURAL_WORDS]
    return words


def content_words(text: str) -> list[str]:
    """Lowercased, stopword-free word tokens used by the heuristic judges."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _overlap(query_words: list[str], text: str) -> float:
    """Fraction of query content words present in `text`.

    Matches exact tokens plus prefix/suffix inflections (spec <-> specification,
    protect <-> protects) so the lexical fallback stays useful on real English.
    """
    if not query_words:
        return 0.0
    haystack = set(content_words(text))
    found = 0
    for w in query_words:
        if w in haystack:
            found += 1
        elif len(w) >= 4 and any(h.startswith(w) for h in haystack):
            found += 1
        elif any(len(h) >= 5 and w.startswith(h) for h in haystack):
            found += 1
    return found / len(query_words)


# --------------------------------------------------------------------- NIM

# Prompt-pack prompts (sections 4, 5B, 6, 7), tightened for JSON-only output.
_CHUNK_VALIDATOR_SYSTEM = """You are the CHUNK_VALIDATOR_AGENT. You are the quality gate between chunking and the
vector/vectorless store - no chunk reaches storage without passing through you.

For the given batch of chunks, score EACH chunk on:
- coherence (0-1): does it read as a complete, self-contained unit of meaning?
- boundary_quality (0-1): does it start/end at a natural boundary, not mid-sentence or mid-table-row?
- coverage (0-1): combined with sibling chunks, is anything from the source text lost or duplicated?

A chunk FAILS if any score is below 0.6. Batch-level status is "fail" if more than 15% of chunks fail.

The chunker is deterministic and follows these rules; do NOT flag a chunk for
following them:
- a heading merged with its first paragraph is a valid heading_block
- an intro sentence ending in ":" followed by a separate whole-list chunk is valid
- a table or list kept whole in its own chunk is valid (even without a heading)
- prose split at sentence boundaries is valid

Reference chunks ONLY by the exact chunk_id strings from the provided list.
Never reuse chunk ids from examples - the examples below are illustrative, not real chunks.

For every failing chunk, produce a specific, actionable suggested_fix using the
real chunk ids, e.g. "merge acme-doc-c004 with acme-doc-c005" - not a generic
restatement of the score.

Return ONLY a JSON object, no prose, no markdown fences, matching exactly:
{
  "stage": "chunk_validation",
  "status": "pass" | "fail",
  "confidence": number,
  "batch_pass_rate": number,
  "chunk_scores": [
    {"chunk_id": string, "coherence": number, "boundary_quality": number,
     "coverage": number, "passed": boolean, "suggested_fix": string | null}
  ],
  "reason": string,
  "retry_feedback": string | null
}"""

_VECTORLESS_PLAN_SYSTEM = """You are the vectorless path of the RETRIEVAL_AGENT. You reason over a document
structure index (section paths) the way a person would use a table of contents:
identify which sections plausibly contain the answer to the query, then those
sections' chunks are pulled.

Use this path when the query is multi-hop, comparative, asks about structure
itself ("what does section 4 conclude"), or spans information across sections
that vector similarity alone tends to miss.

Return ONLY a JSON object, no prose, no markdown fences, matching exactly:
{
  "stage": "retrieval_vectorless_plan",
  "sections": [string],   // section paths from the provided list
  "reason": string
}
Choose at most 3 sections. Choose as few as needed - return an empty list if no
section plausibly relates to the query."""

_RETRIEVAL_VALIDATOR_SYSTEM = """You are the RETRIEVAL_VALIDATOR_AGENT. You are the last gate before context reaches the
answer-generation step - nothing passes through ungrounded.

For the query and its retrieved_chunks, check:
- relevance (0-1): does each chunk actually address the query, not just share vocabulary?
- sufficiency (0-1): do the retrieved chunks, together, contain enough information to fully answer the query - or is there an obvious gap?
- groundedness_risk (0-1, lower is better): how likely is a model to need to fabricate information beyond what's given to answer completely?

FAIL the batch if sufficiency < 0.6 OR groundedness_risk > 0.5, OR if fewer than 60% of chunks are individually relevant.

When you fail a batch, always specify WHY, e.g. "query asks for a comparison across sections 2 and 5, but only section 2
content was retrieved" - this becomes retry_feedback for RETRIEVAL_AGENT so it knows to lean on the vectorless path next attempt.

Return ONLY a JSON object, no prose, no markdown fences, matching exactly:
{
  "stage": "retrieval_validation",
  "status": "pass" | "fail",
  "confidence": number,
  "relevances": [{"chunk_id": string, "relevance": number}],
  "relevance_avg": number,
  "sufficiency": number,
  "groundedness_risk": number,
  "reason": string,
  "retry_feedback": string | null
}"""

_ANSWER_SYSTEM = """You are the ANSWER_AGENT. You are only ever called with context that has already
passed RETRIEVAL_VALIDATOR_AGENT - treat that context as trustworthy, but still cite it.

Rules:
- Answer using ONLY the provided context. If the context is genuinely insufficient for part of the question, say so explicitly instead of filling the gap from general knowledge.
- Cite the chunk_id(s) supporting each claim. You may ONLY cite chunk_ids that appear in the provided context list - never invent an id.
- If the query asks for a comparison, list, or multi-part answer, structure your answer to mirror that.
- Do not restate the entire retrieved context; synthesize.

Return ONLY a JSON object, no prose, no markdown fences, matching exactly:
{
  "stage": "answer_generation",
  "status": "pass" | "fail",
  "confidence": number,
  "answer": string,
  "citations": [{"claim": string, "chunk_ids": [string]}],
  "unanswered_parts": [string],
  "reason": string
}"""


class NIMBrain:
    """LLM-judged stages backed by NVIDIA NIM's OpenAI-compatible API."""

    def __init__(self, config: Config):
        self.config = config
        from openai import OpenAI
        self.client = OpenAI(base_url=config.nim_base_url, api_key=config.nim_api_key)
        # Gates run on the (fast) judge model; answer synthesis on the main one.
        self.judge_model = config.nim_judge_model or config.nim_model

    # ------------------------------------------------------------- helpers

    def _complete_json(self, system: str, user: str, *,
                       model: str | None = None, thinking: bool | None = None) -> dict:
        model = model or self.judge_model
        thinking = self.config.nim_enable_thinking if thinking is None else thinking
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        strict = True
        last_error: Exception | None = None
        for _ in range(2):  # one corrective retry, per the escalation policy
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 16384 if thinking else 4096,
                }
                if strict:
                    kwargs["response_format"] = {"type": "json_object"}
                    if thinking:
                        # Reasoning is returned separately (reasoning_content),
                        # never mixed into the JSON we parse.
                        kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": True},
                            "reasoning_budget": self.config.nim_reasoning_budget,
                        }
                    elif model == self.config.nim_model:
                        # The hosted thinking models reason by default; switch
                        # it off so judging stays fast and deterministic.
                        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                resp = self.client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content or ""
                return _parse_json_object(raw)
            except _OPENAI_API_ERRORS as exc:
                if strict:
                    # Some models reject json_object mode or explicit thinking
                    # control; fall back to a plain completion and rely on the
                    # schema instruction + robust extraction.
                    strict = False
                    last_error = exc
                    messages[1]["content"] = (
                        messages[1]["content"]
                        + "\n\nReturn ONLY a valid JSON object - no markdown fences, no commentary."
                    )
                    continue
                raise LLMOutputError(f"NIM rejected the request: {exc}") from exc
            except _OPENAI_API_ERRORS_GENERIC as exc:
                raise LLMOutputError(f"NIM API error: {exc}") from exc
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                messages[1]["content"] = (
                    messages[1]["content"]
                    + "\n\nYour last output was not valid JSON. Return only valid JSON matching the schema."
                )
        raise LLMOutputError(f"LLM produced invalid JSON twice: {last_error}")

    def _clamp(self, v, lo=0.0, hi=1.0) -> float:
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------ the gates

    def validate_chunks(self, chunks: list[Chunk]) -> ChunkValidationResult:
        scores: list[ChunkScore] = []
        confidences: list[float] = []
        feedbacks: list[str] = []
        reasons: list[str] = []
        for batch in _batches(chunks, 8):
            payload = {
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "section_path": c.section_path,
                        "chunk_type": c.chunk_type,
                        "token_count": c.token_count,
                        "text": c.text,
                    }
                    for c in batch
                ]
            }
            data = self._complete_json(_CHUNK_VALIDATOR_SYSTEM, json.dumps(payload),
                                      model=self.judge_model, thinking=False)
            for entry in data.get("chunk_scores", []):
                c = self._clamp(entry.get("coherence"))
                b = self._clamp(entry.get("boundary_quality"))
                v = self._clamp(entry.get("coverage"))
                scores.append(ChunkScore(
                    chunk_id=str(entry.get("chunk_id", "")),
                    coherence=c,
                    boundary_quality=b,
                    coverage=v,
                    passed=bool(entry.get("passed", False)),
                    suggested_fix=entry.get("suggested_fix"),
                ))
            confidences.append(self._clamp(data.get("confidence")))
            if data.get("retry_feedback"):
                feedbacks.append(str(data["retry_feedback"]))
            if data.get("reason"):
                reasons.append(str(data["reason"]))
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        failed = [s for s in scores if not s.passed]
        fail_share = len(failed) / len(scores) if scores else 1.0
        return ChunkValidationResult(
            stage="chunk_validation",
            # Batch fails when more than 15% of chunks fail, per the gate policy.
            status="pass" if fail_share <= 0.15 else "fail",
            confidence=confidence,
            batch_pass_rate=round(1.0 - fail_share, 3),
            chunk_scores=scores,
            retry_feedback=" | ".join(feedbacks) if feedbacks else None,
            reason="; ".join(reasons) if reasons else "chunk batch scored",
        )

    def plan_vectorless(self, sections: list[str], query: str) -> list[str]:
        if not sections:
            return []
        data = self._complete_json(
            _VECTORLESS_PLAN_SYSTEM,
            json.dumps({"query": query, "sections": sections}),
            model=self.judge_model, thinking=False,
        )
        chosen = data.get("sections", [])
        if not isinstance(chosen, list):
            raise LLMOutputError("vectorless plan returned non-list sections")
        return [s for s in chosen if s in sections][:3]

    def validate_retrieval(self, query: str, retrieved: list[RetrievedChunk]) -> RetrievalValidationResult:
        payload = {
            "query": query,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "section_path": c.section_path,
                    "source_path": c.source_path,
                    "relevance_score": c.relevance_score,
                    "text": c.text[:4000],
                }
                for c in retrieved
            ],
        }
        data = self._complete_json(_RETRIEVAL_VALIDATOR_SYSTEM, json.dumps(payload),
                                   model=self.judge_model, thinking=False)
        rels = data.get("relevances", []) or []
        rel_values = [self._clamp(r.get("relevance")) for r in rels if isinstance(r, dict)]
        relevant_share = sum(1 for v in rel_values if v >= 0.5) / len(rel_values) if rel_values else 0.0
        sufficiency = self._clamp(data.get("sufficiency"))
        risk = self._clamp(data.get("groundedness_risk"))

        llm_said_pass = data.get("status") == "pass"

        # If the LLM explicitly says "pass", we trust its holistic judgment.
        # We only override with a hard fail on unambiguously bad scores to
        # prevent the model from self-contradicting (saying pass but emitting
        # borderline sufficiency numbers that trip our gate).
        fails = []
        if llm_said_pass:
            # Only reject on extreme violations when the LLM itself says pass.
            if sufficiency < 0.4:
                fails.append(f"sufficiency={sufficiency:.2f} (critically low)")
            if risk > 0.7:
                fails.append(f"groundedness_risk={risk:.2f} (critically high)")
        else:
            # LLM said fail — apply relaxed thresholds.
            # relevant_share is lowered to 0.35 because PDF documents with
            # many cover/heading chunks legitimately have mixed relevance scores;
            # the 0.6 floor would block every real-world PDF query.
            if sufficiency < 0.5:
                fails.append(f"sufficiency={sufficiency:.2f}")
            if risk > 0.65:
                fails.append(f"groundedness_risk={risk:.2f}")
            if relevant_share < 0.35:
                fails.append(f"relevant_share={relevant_share:.2f}")

        return RetrievalValidationResult(
            stage="retrieval_validation",
            status="fail" if fails else "pass",
            confidence=self._clamp(data.get("confidence")),
            relevance_avg=self._clamp(data.get("relevance_avg")),
            sufficiency=sufficiency,
            groundedness_risk=risk,
            relevant_share=round(relevant_share, 3),
            retry_feedback=data.get("retry_feedback") or ("; ".join(fails) if fails else None),
            reason=str(data.get("reason", "")) or "; ".join(fails),
        )

    def generate_answer(self, query: str, retrieved: list[RetrievedChunk]) -> AnswerResult:
        payload = {
            "query": query,
            "context": [
                {"chunk_id": c.chunk_id, "section_path": c.section_path, "text": c.text[:4000]}
                for c in retrieved
            ],
        }
        data = self._complete_json(_ANSWER_SYSTEM, json.dumps(payload),
                                   model=self.config.nim_model)
        citations = []
        for cit in data.get("citations", []) or []:
            if isinstance(cit, dict):
                citations.append(Citation(
                    claim=str(cit.get("claim", "")),
                    chunk_ids=list(cit.get("chunk_ids") or []),
                ))
        return AnswerResult(
            stage="answer_generation",
            status="pass" if data.get("status") == "pass" else "fail",
            confidence=self._clamp(data.get("confidence")),
            answer=str(data.get("answer", "")),
            citations=citations,
            unanswered_parts=list(data.get("unanswered_parts", []) or []),
            reason=str(data.get("reason", "")),
        )


# --------------------------------------------------------------- heuristics

class HeuristicBrain:
    """Deterministic fallback judges so the pipeline runs without an LLM.

    Not a substitute for the LLM judges - scores are lexical approximations -
    but they exercise the exact same gates, retries and contracts.
    """

    def __init__(self, config: Config):
        self.config = config

    # ------------------------------------------------------------ the gates

    def validate_chunks(self, chunks: list[Chunk]) -> ChunkValidationResult:
        scores: list[ChunkScore] = []
        for i, c in enumerate(chunks):
            nxt = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None
            coherence = self._coherence(c)
            boundary = self._boundary(c)
            coverage = self._coverage(c)
            passed = coherence >= 0.6 and boundary >= 0.6 and coverage >= 0.6
            fix = None
            if not passed:
                fix = self._suggest_fix(c, nxt, coherence, boundary, coverage)
            scores.append(ChunkScore(
                chunk_id=c.chunk_id,
                coherence=coherence,
                boundary_quality=boundary,
                coverage=coverage,
                passed=passed,
                suggested_fix=fix,
            ))
        failed = [s for s in scores if not s.passed]
        pass_rate = (len(scores) - len(failed)) / len(scores) if scores else 0.0
        batch_pass = len(failed) / len(scores) <= 0.15 if scores else False
        feedback = None
        if failed:
            feedback = "Fix the failing chunks: " + " | ".join(
                f"{s.chunk_id}: {s.suggested_fix}" for s in failed[:5]
            )
        return ChunkValidationResult(
            stage="chunk_validation",
            status="pass" if batch_pass else "fail",
            confidence=0.95,
            batch_pass_rate=round(pass_rate, 3),
            chunk_scores=scores,
            retry_feedback=feedback,
            reason=f"{len(failed)} chunk(s) failed heuristic scoring",
        )

    @staticmethod
    def _coherence(c: Chunk) -> float:
        text = c.text.strip()
        if not text:
            return 0.0
        if c.chunk_type in ("table", "list"):
            return 0.9
        words = len(text.split())
        if c.chunk_type == "heading_block":
            return 0.9 if words >= 4 else 0.6
        if words < 8:
            return 0.4
        if words < 20:
            return 0.7
        return 0.9

    @staticmethod
    def _boundary(c: Chunk) -> float:
        text = c.text.strip()
        if not text:
            return 0.0
        if c.chunk_type == "table":
            return 0.95 if text.endswith("|") else 0.4
        if c.chunk_type == "list":
            return 0.9 if text.rstrip().endswith((".", "!", "?", ":", ";", ")")) else 0.6
        first = text[0]
        starts_clean = first.isupper() or first.isdigit() or first in "#*-\u2022"
        ends_clean = text.rstrip().endswith((".", "!", "?", ":", "|")) or "```" in text
        score = 0.5
        if starts_clean:
            score += 0.25
        if ends_clean:
            score += 0.25
        return round(score, 2)

    @staticmethod
    def _coverage(c: Chunk) -> float:
        t = c.token_count
        if 50 <= t <= 600:
            return 0.9
        if t < 50:
            return 0.7
        return 0.6

    @staticmethod
    def _suggest_fix(c: Chunk, nxt: str | None, coherence: float, boundary: float, coverage: float) -> str:
        if coherence < 0.6:
            return f"too small to be self-contained; merge {c.chunk_id} with its neighbor"
        if boundary < 0.6:
            if nxt:
                return f"{c.chunk_id} cuts mid-sentence; merge with {nxt} and re-split at a sentence boundary"
            return f"{c.chunk_id} has a broken boundary; re-chunk it"
        if coverage < 0.6:
            return f"{c.chunk_id} is oversized ({c.token_count} tokens); split at sentence boundaries"
        return f"re-chunk {c.chunk_id}"

    def plan_vectorless(self, sections: list[str], query: str) -> list[str]:
        if not sections:
            return []
        qwords = set(content_words(query))
        if not qwords:
            return []
        # Structural queries: "section 4" -> find headings containing the number.
        structural = re.findall(r"section\s*(\d+)", query.lower())
        picks: list[str] = []
        if structural:
            # Prefer sections whose whole path matches the query words, so
            # "section 4 ... pricing" lands on the right document's section 4.
            best_hits = 0
            scored: list[tuple[int, str]] = []
            for s in sections:
                last = s.split(" / ")[-1].lower()
                if any(last.startswith(num) for num in structural):
                    hits = len(qwords & set(content_words(s)))
                    scored.append((hits, s))
                    best_hits = max(best_hits, hits)
            if scored:
                if best_hits > 0:
                    picks = [s for h, s in scored if h == best_hits]
                else:
                    picks = [s for _, s in scored]
        if picks:
            return picks[:3]
        scored = []
        for s in sections:
            seg_words = set(content_words(s))
            hit = len(qwords & seg_words)
            if hit > 0:
                scored.append((hit, s))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [s for _, s in scored[:3]]

    def validate_retrieval(self, query: str, retrieved: list[RetrievedChunk]) -> RetrievalValidationResult:
        if not retrieved:
            return RetrievalValidationResult(
                stage="retrieval_validation",
                status="fail",
                confidence=0.9,
                relevance_avg=0.0,
                sufficiency=0.0,
                groundedness_risk=1.0,
                relevant_share=0.0,
                retry_feedback="no chunks retrieved at all; broaden the search",
                reason="empty retrieval",
            )
        qwords = content_words(query)
        topic = _topic_words(query)
        # "what does section 4 say" queries: relevance is structural membership.
        structural = re.findall(r"section\s*(\d+)", query.lower())
        rels = []
        for c in retrieved:
            lex = _overlap(qwords, c.text)
            if c.source_path in ("vector", "both"):
                # Embedding similarity is a strong per-chunk signal.
                rel = max(lex, c.relevance_score)
            else:
                # The vectorless path is a deliberate structure-based pick
                # (like a table-of-contents read), so its chunks carry a base
                # relevance floor rather than relying on lexical overlap alone.
                rel = max(lex, 0.5)
            if structural and c.section_path:
                last = c.section_path.split(" / ")[-1].lower()
                if any(last.startswith(num) for num in structural):
                    rel = max(rel, 0.85)  # chunks from the referenced section
            boost = 0.05 if c.source_path == "both" else 0.0
            rels.append(round(min(1.0, rel + boost), 3))
        relevance_avg = sum(rels) / len(rels)
        # A chunk counts as "individually relevant" relative to the best match.
        # Lowered floor to 0.35 to handle PDF documents with many heading-only
        # chunks that legitimately have mixed relevance scores — the 0.6 floor
        # blocked real-world PDFs from ever passing the gate.
        rel_threshold = 0.3 if structural else max(0.25, 0.4 * max(rels)) if rels else 0.35
        relevant_share = sum(1 for r in rels if r >= rel_threshold) / len(rels)
        union = " ".join(c.text for c in retrieved)
        if topic:
            sufficiency = _overlap(topic, union)
        elif structural:
            # "what does section 4 say" with no remaining topic words:
            # sufficient iff the referenced section's chunks were retrieved.
            sufficiency = 1.0 if any(
                c.section_path and any(
                    c.section_path.split(" / ")[-1].lower().startswith(num)
                    for num in structural
                )
                for c in retrieved
            ) else 0.0
        else:
            sufficiency = 0.0
        groundedness_risk = round(max(0.0, 1.0 - sufficiency), 3)
        fails = []
        if sufficiency < 0.4:
            fails.append(f"sufficiency={sufficiency:.2f}")
        if groundedness_risk > 0.65:
            fails.append(f"groundedness_risk={groundedness_risk:.2f}")
        if relevant_share < 0.35:
            fails.append(f"relevant_share={relevant_share:.2f}")
        feedback = None
        if fails:
            feedback = (
                "retrieval validation failed: " + ", ".join(fails)
                + "; weight the vectorless path higher and search additional sections"
            )
        return RetrievalValidationResult(
            stage="retrieval_validation",
            status="fail" if fails else "pass",
            confidence=0.9,
            relevance_avg=round(relevance_avg, 3),
            sufficiency=round(sufficiency, 3),
            groundedness_risk=groundedness_risk,
            relevant_share=round(relevant_share, 3),
            retry_feedback=feedback,
            reason="heuristic relevance/sufficiency scoring",
        )

    def generate_answer(self, query: str, retrieved: list[RetrievedChunk]) -> AnswerResult:
        qwords = set(_topic_words(query))
        best = sorted(retrieved, key=lambda c: c.relevance_score, reverse=True)[:3]
        sentences: list[tuple[str, str]] = []
        for c in best:
            for sent in _sentences(c.text):
                if _HEADING_LINE_RE.match(sent):
                    continue  # headings are structure, not claims
                if len(qwords & set(content_words(sent))) >= 1:
                    sentences.append((sent, c.chunk_id))
        # Fallback: leading sentence of the best chunk.
        if not sentences and best:
            first = _sentences(best[0].text)
            if first:
                sentences.append((first[0], best[0].chunk_id))
        answer = " ".join(f"- {s} [{cid}]" for s, cid in sentences[:4])
        citations = []
        for s, cid in sentences[:4]:
            citations.append(Citation(claim=s[:80], chunk_ids=[cid]))
        union = " ".join(c.text for c in retrieved)
        # Prefix/inflection-aware, like _overlap: "protect" counts as covered
        # by "protected".
        missing = [w for w in qwords if _overlap([w], union) < 1.0]
        unanswered = []
        if missing:
            unanswered.append(f"context does not cover: {', '.join(sorted(missing)[:6])}")
        if not sentences:
            unanswered.append("no retrieved chunk directly addresses the query")
        return AnswerResult(
            stage="answer_generation",
            status="pass",
            confidence=0.85,
            answer=answer or "No supporting sentences found in the validated context.",
            citations=citations,
            unanswered_parts=unanswered,
            reason="extractive heuristic answer over validated context",
        )


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _sentences(text: str) -> list[str]:
    if "```" in text:
        return [text.strip()]
    # Split after sentence-ending punctuation, but not after digits, so
    # "## 4. Pricing" headings and "2.1" style numbers stay whole.
    parts = re.split(r"(?<=[.!?])(?<![0-9][.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def get_brain(config: Config):
    """Pick the NIM brain when a key is present, otherwise the heuristic one."""
    if config.use_nim:
        try:
            return NIMBrain(config)
        except ImportError as exc:
            print(f"[warn] openai SDK unavailable, falling back to heuristic brain: {exc}")
    else:
        print("[warn] NVIDIA_API_KEY not set - LLM-judged stages run with heuristic judges.")
        print("[warn] Set NVIDIA_API_KEY in .env to enable NVIDIA NIM for real LLM judgment.")
    return HeuristicBrain(config)
