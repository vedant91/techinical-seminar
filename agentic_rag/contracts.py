"""Strict typed contracts for every stage hand-off in the pipeline.

The JSON shapes mirror the prompt pack exactly so that each agent's output can be
logged, validated, and machine-parsed end-to-end.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------- stage results

@dataclass
class StageResult:
    stage: str
    status: str          # "pass" | "fail"
    confidence: float    # 0.0 - 1.0
    reason: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in dataclasses.asdict(self).items()}
        d.pop("block_tree", None)
        return d


# ------------------------------------------------------------------ ingestion

@dataclass
class OutlineEntry:
    level: int
    heading: str
    start_offset: int


@dataclass
class TableRegion:
    location: str
    markdown: str


@dataclass
class UnparsedRegion:
    location: str
    reason: str


@dataclass
class Document:
    doc_id: str
    title: str
    source_type: str
    outline: list = field(default_factory=list)          # list[OutlineEntry]
    normalized_text: str = ""
    tables: list = field(default_factory=list)           # list[TableRegion]
    unparsed_regions: list = field(default_factory=list) # list[UnparsedRegion]
    # Internal (not part of the strict JSON contract): structured block tree used
    # by the chunker so it never has to re-parse flattened text.
    block_tree: list = field(default_factory=list, repr=False)


@dataclass
class IngestionResult(StageResult):
    document: Document | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("block_tree", None)
        if d.get("document"):
            d["document"].pop("block_tree", None)
        return d


# ------------------------------------------------------------------- chunking

@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_path: str
    token_count: int
    chunk_type: str       # "prose" | "table" | "list" | "heading_block"


@dataclass
class ChunkingResult(StageResult):
    chunks: list = field(default_factory=list)           # list[Chunk]


# ------------------------------------------------------------ chunk validation

@dataclass
class ChunkScore:
    chunk_id: str
    coherence: float
    boundary_quality: float
    coverage: float
    passed: bool
    suggested_fix: str | None = None


@dataclass
class ChunkValidationResult(StageResult):
    batch_pass_rate: float = 0.0
    chunk_scores: list = field(default_factory=list)     # list[ChunkScore]
    retry_feedback: str | None = None


# ------------------------------------------------------------------ retrieval

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    section_path: str
    source_path: str      # "vector" | "vectorless" | "both"
    relevance_score: float


@dataclass
class RetrievalResult(StageResult):
    query: str = ""
    retrieved_chunks: list = field(default_factory=list) # list[RetrievedChunk]


# --------------------------------------------------------- retrieval validation

@dataclass
class RetrievalValidationResult(StageResult):
    relevance_avg: float = 0.0
    sufficiency: float = 0.0
    groundedness_risk: float = 0.0
    relevant_share: float = 0.0          # internal: fraction of chunks with relevance >= 0.5
    retry_feedback: str | None = None


# --------------------------------------------------------------------- answer

@dataclass
class Citation:
    claim: str
    chunk_ids: list


@dataclass
class AnswerResult(StageResult):
    answer: str = ""
    citations: list = field(default_factory=list)        # list[Citation]
    unanswered_parts: list = field(default_factory=list)


# -------------------------------------------------------------- pipeline-level

@dataclass
class StageLogEntry:
    stage: str
    status: str
    confidence: float
    retries: int
    timestamp: str


@dataclass
class PipelineResult:
    task_id: str
    final_status: str            # "success" | "escalated_to_human" | "failed"
    stages: list = field(default_factory=list)           # list[StageLogEntry]
    result: Any = None
    escalation_reason: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
