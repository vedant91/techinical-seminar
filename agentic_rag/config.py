"""Central configuration for the Agentic RAG pipeline.

All settings come from environment variables or a `.env` file in the repo root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load the repo-root .env before the Config defaults are evaluated.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Config:
    # NVIDIA NIM (OpenAI-compatible). Key from https://build.nvidia.com
    nim_api_key: str = field(default_factory=lambda: _env("NVIDIA_API_KEY", ""))
    nim_base_url: str = field(default_factory=lambda: _env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    nim_model: str = field(default_factory=lambda: _env("NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"))
    # Judge model for the quality gates (chunk validation, vectorless plan,
    # retrieval validation). Empty = same as nim_model. A fast model keeps the
    # retry loop cheap; the big model is reserved for answer synthesis.
    nim_judge_model: str = field(default_factory=lambda: _env("NIM_JUDGE_MODEL", ""))
    # Optional reasoning mode: the model thinks before answering, and the
    # reasoning is returned separately (never mixed into the JSON).
    nim_enable_thinking: bool = field(default_factory=lambda: _env("NIM_ENABLE_THINKING", "0") == "1")
    nim_reasoning_budget: int = field(default_factory=lambda: int(_env("NIM_REASONING_BUDGET", "16384")))

    # PostgreSQL + pgvector
    pg_dsn: str = field(default_factory=lambda: _env("PG_DSN", "postgresql://rag:rag@localhost:5432/ragdb"))

    # Open-source local embedding model
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # Pipeline tuning
    max_retries: int = field(default_factory=lambda: int(_env("MAX_RETRIES", "2")))
    retrieval_top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", "8")))
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", "logs"))

    @property
    def use_nim(self) -> bool:
        """True when a NIM API key is available; otherwise LLM stages run heuristically."""
        return bool(self.nim_api_key)
