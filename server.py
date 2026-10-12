"""
FastAPI server that bridges the Agentic RAG backend to the frontend.
Saves uploaded files to a temp dir, runs the real pipeline, and returns JSON.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agentic_rag import AgenticRAG

# ── Bootstrap ────────────────────────────────────────────────────────────────

app = FastAPI(title="Agentic RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton pipeline (loads embedder + connects to PG once on startup)
_rag: AgenticRAG | None = None


def get_rag() -> AgenticRAG:
    global _rag
    if _rag is None:
        _rag = AgenticRAG()
    return _rag


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses / custom objects to plain dicts / lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _make_serializable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if k != "block_tree"}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    return obj


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    """Return store type and how many chunks are currently indexed."""
    rag = get_rag()
    store = rag.store
    store_type = type(store).__name__   # PgVectorStore or InMemoryStore
    try:
        if hasattr(store, "records"):          # InMemoryStore
            chunk_count = len(store.records)
        elif hasattr(store, "conn"):           # PgVectorStore
            row = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = int(row[0]) if row else 0
        else:
            chunk_count = -1
    except Exception:
        chunk_count = -1
    return {
        "store_type": store_type,
        "chunk_count": chunk_count,
        "ready": chunk_count > 0,
    }


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Accept a .md or .txt file, run it through the full ingestion + chunking
    + chunk-validation pipeline, and return the structured result.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".md", ".txt", ".markdown", ".pdf"):
        raise HTTPException(400, "Only .md, .txt, and .pdf files are supported.")

    rag = get_rag()

    # Write to a temp file so the existing file-based agents work unchanged
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                     mode="wb") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = rag.ingest_document(tmp_path)
    finally:
        os.unlink(tmp_path)

    payload = _make_serializable(result)
    return payload


@app.post("/api/ask")
async def ask(query: str = Form(...)):
    """
    Run the hybrid retrieval + validation + answer pipeline for a query.
    The document must have been ingested first (embeddings are persisted in
    the vector store for the lifetime of this server process).
    """
    if not query.strip():
        raise HTTPException(400, "Query must not be empty.")

    rag = get_rag()
    result = rag.ask(query)
    payload = _make_serializable(result)
    return payload


# ── Serve frontend ────────────────────────────────────────────────────────────
# Mount AFTER the /api routes so FastAPI matches API paths first.
# html=True makes StaticFiles serve index.html for "/" automatically and
# resolves relative asset paths (style.css, script.js) from root correctly.

FRONTEND_DIR = Path(__file__).parent / "hybrid_search_viz"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
