"""High-level facade: build the pipeline and talk to it.

    rag = AgenticRAG()
    rag.ingest_document("docs/spec.md")
    result = rag.ask("What does section 4 conclude?")
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .contracts import PipelineResult
from .embeddings import Embedder
from .llm import get_brain
from .orchestrator import Orchestrator
from .store import InMemoryStore, PgVectorStore, VectorStore


class AgenticRAG:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.embedder = Embedder(self.config.embedding_model)
        # Load the embedding model up front: if it fails, fail loudly here with
        # a clear message instead of a confusing cascade later.
        try:
            dim = self.embedder.dim
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the embedding model '{self.config.embedding_model}'. "
                f"This usually means torch and numpy versions are incompatible "
                f"(torch >= 2.4 is required for numpy 2.x). "
                f"Try: pip install -U torch"
            ) from exc
        self.brain = get_brain(self.config)
        self.store = self._make_store(dim)
        self.orchestrator = Orchestrator(
            store=self.store,
            embedder=self.embedder,
            brain=self.brain,
            max_retries=self.config.max_retries,
            top_k=self.config.retrieval_top_k,
            log_dir=self.config.log_dir,
        )

    def _make_store(self, dim: int) -> VectorStore:
        try:
            store = PgVectorStore(self.config.pg_dsn, dim)
            print(f"[store] connected to pgvector ({self.config.pg_dsn})")
            return store
        except Exception as exc:
            print(f"[warn] pgvector unreachable ({exc.__class__.__name__}).")
            print("[warn] Falling back to the in-memory store - start Docker Desktop and")
            print("[warn] run 'docker compose up -d' to use PostgreSQL + pgvector.")
            return InMemoryStore(dim)

    # ------------------------------------------------------------ public API

    def ingest_document(self, path: str | Path) -> PipelineResult:
        return self.orchestrator.ingest(str(path))

    def ask(self, query: str) -> PipelineResult:
        return self.orchestrator.ask(query)

    # ------------------------------------------------------------- reporting

    @staticmethod
    def print_result(result: PipelineResult, verbose: bool = False) -> None:
        print(f"\nTask {result.task_id}  final_status={result.final_status}")
        print("-" * 76)
        print(f"{'stage':<22}{'status':<10}{'conf':<8}{'retries':<8}")
        for entry in result.stages:
            print(f"{entry.stage:<22}{entry.status:<10}{entry.confidence:<8.3f}{entry.retries:<8}")
        print("-" * 76)

        if result.escalation_reason:
            print(f"escalation_reason: {result.escalation_reason}")
        if result.result is None:
            return

        if isinstance(result.result, dict):
            payload = result.result.get("answer")
            ret = result.result.get("retrieval")
            if payload is not None:
                _print_answer(payload, ret, verbose)
            elif "error" in result.result:
                print(f"result: {json.dumps(result.result, ensure_ascii=False)}")
            else:
                print(f"result: {json.dumps(result.result, ensure_ascii=False)}")


def _print_answer(payload, ret, verbose: bool) -> None:
    print("\nANSWER:")
    print(payload.answer)
    if payload.citations:
        print("\nCITATIONS:")
        for cit in payload.citations:
            print(f"  - {cit.claim[:90]} -> {cit.chunk_ids}")
    if payload.unanswered_parts:
        print("\nUNANSWERED PARTS (honest gaps, not fabrications):")
        for part in payload.unanswered_parts:
            print(f"  - {part}")
    if ret is not None:
        counts = {"vector": 0, "vectorless": 0, "both": 0}
        for c in ret.retrieved_chunks:
            counts[c.source_path] += 1
        print(f"\nRETRIEVAL PATHS: vector={counts['vector']}, "
              f"vectorless={counts['vectorless']}, both={counts['both']}")
        print(f"retrieval reason: {ret.reason}")
        if verbose:
            print("\nRETRIEVED CHUNKS:")
            for c in ret.retrieved_chunks:
                print(f"  [{c.source_path:<10} {c.relevance_score:.3f}] {c.chunk_id} "
                      f"({c.section_path}): {c.text[:120].strip()!r}")
