"""The ORCHESTRATOR.

Routes work between the six specialist agents and enforces the quality gates.
It never reads documents or answers questions itself; it decides what happens
on failure - retry with the validator's feedback, quarantine, or escalate to a
human. A visible failure is always better than a silent one.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .agents import answer, chunk_validator, chunking, ingestion, retrieval, retrieval_validator
from .contracts import (
    Chunk,
    ChunkValidationResult,
    PipelineResult,
    StageLogEntry,
)
from .llm import LLMOutputError


class Orchestrator:
    def __init__(self, store, embedder, brain, max_retries: int = 2,
                 top_k: int = 8, log_dir: str = "logs"):
        self.store = store
        self.embedder = embedder
        self.brain = brain
        self.max_retries = max_retries
        self.top_k = top_k
        self.log_path = Path(log_dir) / "stage_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stage_log: list[StageLogEntry] = []

    # ------------------------------------------------------------ public API

    def ingest(self, file_path: str) -> PipelineResult:
        task_id = uuid.uuid4().hex[:8]
        self._stage_log = []

        # Stage 1: ingestion (deterministic; a fail here is a hard stop).
        ing = ingestion.ingest_file(file_path)
        self._log("ingestion", ing.status, ing.confidence, 0)
        if ing.status != "pass" or ing.document is None:
            return self._result(task_id, "failed", None, ing.reason)

        doc = ing.document
        try:
            chunks, validation, retries = self._chunk_with_gates(doc)
        except LLMOutputError as exc:
            self._log("chunk_validation", "fail", 0.0, self.max_retries)
            return self._result(task_id, "escalated_to_human", None,
                                f"chunk judge produced malformed output: {exc}")

        if validation.status != "pass":
            # Retries exhausted: quarantine failing chunks, keep the passing
            # ones so one bad section never blocks the whole document.
            passing, failing = self._split_by_score(chunks, validation)
            if passing:
                self._store_chunks(doc.doc_id, passing)
                self._quarantine(doc.doc_id, failing, validation)
                self._log("quarantine", "pass", 0.9, retries)
                return self._result(task_id, "success",
                                    {"document": doc.doc_id, "chunks_stored": len(passing),
                                     "chunks_quarantined": len(failing)},
                                    f"stored {len(passing)} chunk(s); {len(failing)} quarantined for human review")
            return self._result(task_id, "escalated_to_human", None,
                                f"all {len(chunks)} chunks failed validation after {retries} retries")

        self._store_chunks(doc.doc_id, chunks)
        return self._result(task_id, "success",
                            {"document": doc.doc_id, "chunks_stored": len(chunks),
                             "chunks_quarantined": 0},
                            f"stored {len(chunks)} validated chunk(s)")

    def ask(self, query: str) -> PipelineResult:
        task_id = uuid.uuid4().hex[:8]
        self._stage_log = []

        try:
            ret, validation, retries = self._retrieve_with_gates(query)
        except LLMOutputError as exc:
            self._log("retrieval_validation", "fail", 0.0, self.max_retries)
            return self._result(task_id, "failed", None,
                                f"retrieval judge produced malformed output: {exc}")

        if validation.status != "pass":
            # Never pass ungrounded context to ANSWER_AGENT. Fail visibly.
            return self._result(
                task_id, "failed",
                {"error": "insufficient_context"},
                f"retrieval could not satisfy the validator after {retries} retry attempt(s): "
                f"{validation.retry_feedback or validation.reason}",
            )

        try:
            ans = answer.generate_answer(query, ret.retrieved_chunks, self.brain)
        except LLMOutputError as exc:
            self._log("answer_generation", "fail", 0.0, 0)
            return self._result(task_id, "failed", None,
                                f"answer agent produced malformed output: {exc}")
        self._log("answer_generation", ans.status, ans.confidence, 0)

        return self._result(
            task_id, "success" if ans.status == "pass" else "failed",
            {"answer": ans, "retrieval": ret}, ans.reason,
        )

    # ----------------------------------------------------- stage internals

    def _chunk_with_gates(self, doc):
        retries = 0
        feedback = None
        chunks_result = chunking.chunk_document(doc)
        self._log("chunking", chunks_result.status, chunks_result.confidence, 0)

        validation = chunk_validator.validate_chunks(chunks_result.chunks, self.brain)
        self._log("chunk_validation", validation.status, validation.confidence, 0)
        while validation.status != "pass" and retries < self.max_retries:
            retries += 1
            feedback = validation.retry_feedback
            chunks_result = chunking.chunk_document(doc, retry_feedback=feedback)
            self._log("chunking", chunks_result.status, chunks_result.confidence, retries)
            validation = chunk_validator.validate_chunks(chunks_result.chunks, self.brain)
            self._log("chunk_validation", validation.status, validation.confidence, retries)
        return chunks_result.chunks, validation, retries

    def _retrieve_with_gates(self, query):
        retries = 0
        feedback = None
        ret = retrieval.retrieve(query, self.store, self.embedder, self.brain,
                                 self.top_k, retry_feedback=feedback)
        self._log("retrieval", ret.status, ret.confidence, 0)
        if not ret.retrieved_chunks:
            # Empty retrieval: still run the gate so the failure is documented.
            validation = retrieval_validator.validate_retrieval(query, [], self.brain)
            self._log("retrieval_validation", validation.status, validation.confidence, 0)
            return ret, validation, retries
        validation = retrieval_validator.validate_retrieval(query, ret.retrieved_chunks, self.brain)
        self._log("retrieval_validation", validation.status, validation.confidence, 0)
        while validation.status != "pass" and retries < self.max_retries:
            retries += 1
            feedback = validation.retry_feedback
            ret = retrieval.retrieve(query, self.store, self.embedder, self.brain,
                                     self.top_k, retry_feedback=feedback)
            self._log("retrieval", ret.status, ret.confidence, retries)
            validation = retrieval_validator.validate_retrieval(query, ret.retrieved_chunks, self.brain)
            self._log("retrieval_validation", validation.status, validation.confidence, retries)
        return ret, validation, retries

    def _store_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed_texts([c.text for c in chunks])
        self.store.upsert_doc(doc_id, chunks, vectors)

    @staticmethod
    def _split_by_score(chunks: list[Chunk], validation: ChunkValidationResult):
        passed_ids = {s.chunk_id for s in validation.chunk_scores if s.passed}
        passing = [c for c in chunks if c.chunk_id in passed_ids]
        failing = [c for c in chunks if c.chunk_id not in passed_ids]
        return passing, failing

    def _quarantine(self, doc_id: str, failing: list[Chunk],
                    validation: ChunkValidationResult) -> None:
        fixes = {s.chunk_id: s.suggested_fix for s in validation.chunk_scores}
        qpath = self.log_path.parent / "quarantine.jsonl"
        with qpath.open("a", encoding="utf-8") as fh:
            for c in failing:
                fh.write(json.dumps({
                    "timestamp": _now(),
                    "doc_id": doc_id,
                    "chunk_id": c.chunk_id,
                    "suggested_fix": fixes.get(c.chunk_id),
                    "text": c.text[:300],
                }) + "\n")

    # ------------------------------------------------------------ bookkeeping

    def _log(self, stage: str, status: str, confidence: float, retries: int) -> None:
        entry = StageLogEntry(stage=stage, status=status,
                              confidence=round(float(confidence), 3),
                              retries=retries, timestamp=_now())
        self._stage_log.append(entry)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__) + "\n")

    def _result(self, task_id: str, final_status: str, result, escalation_reason) -> PipelineResult:
        return PipelineResult(
            task_id=task_id,
            final_status=final_status,
            stages=list(self._stage_log),
            result=result,
            escalation_reason=escalation_reason,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
