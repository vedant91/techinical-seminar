"""Agentic RAG: a 6-agent, self-validating hybrid retrieval pipeline.

Pipeline: Ingestion -> Chunking -> Chunk Validation -> Hybrid Retrieval
(vector + vectorless) -> Retrieval Validation -> Answer Generation,
coordinated by an Orchestrator that retries failed stages instead of letting
them fail silently.
"""
from .config import Config
from .pipeline import AgenticRAG

__all__ = ["AgenticRAG", "Config"]
