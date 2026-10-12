"""Vector store for Path A (vector retrieval) with two interchangeable backends.

- `PgVectorStore` - PostgreSQL + pgvector (production path; see docker-compose.yml).
- `InMemoryStore`  - numpy fallback used when the database is unreachable, so the
  seminar demo never blocks on infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import Chunk


@dataclass
class StoredChunk:
    doc_id: str
    chunk_id: str
    text: str
    section_path: str
    chunk_type: str
    token_count: int
    score: float = 0.0


class VectorStore:
    """Common interface for both backends."""

    def upsert_doc(self, doc_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[StoredChunk]:
        raise NotImplementedError

    def chunks_in_section(self, section_path: str) -> list[StoredChunk]:
        raise NotImplementedError

    def get_sections(self) -> list[str]:
        raise NotImplementedError


class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, connect_timeout=5)
        register_vector(self.conn)
        self.dim = dim
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                doc_id       TEXT NOT NULL,
                chunk_id     TEXT PRIMARY KEY,
                text         TEXT NOT NULL,
                section_path TEXT NOT NULL,
                chunk_type   TEXT NOT NULL,
                token_count  INT  NOT NULL,
                embedding    vector({int(dim)})
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        self.conn.commit()

    # ------------------------------------------------------------- interface

    def upsert_doc(self, doc_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        with self.conn.transaction():
            self.conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            for c, emb in zip(chunks, embeddings):
                self.conn.execute(
                    "INSERT INTO chunks (doc_id, chunk_id, text, section_path, chunk_type, token_count, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (doc_id, c.chunk_id, c.text, c.section_path, c.chunk_type, c.token_count, emb),
                )

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[StoredChunk]:
        rows = self.conn.execute(
            "SELECT doc_id, chunk_id, text, section_path, chunk_type, token_count,"
            "       1 - (embedding <=> %s) AS score"
            " FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (query_embedding, query_embedding, top_k),
        ).fetchall()
        return [self._row(r) for r in rows]

    def chunks_in_section(self, section_path: str) -> list[StoredChunk]:
        rows = self.conn.execute(
            "SELECT doc_id, chunk_id, text, section_path, chunk_type, token_count, 0.0 AS score"
            " FROM chunks WHERE section_path = %s OR strpos(section_path, %s) = 1",
            (section_path, section_path + " / "),
        ).fetchall()
        return [self._row(r) for r in rows]

    def get_sections(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT section_path FROM chunks ORDER BY section_path").fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _row(r) -> StoredChunk:
        return StoredChunk(
            doc_id=r[0], chunk_id=r[1], text=r[2], section_path=r[3],
            chunk_type=r[4], token_count=r[5], score=float(r[6] or 0.0),
        )

    def close(self) -> None:
        self.conn.close()


class InMemoryStore(VectorStore):
    """numpy-backed fallback: cosine similarity over normalized embeddings."""

    def __init__(self, dim: int):
        self.dim = dim
        self.records: list[StoredChunk] = []
        self.matrix: np.ndarray | None = None

    def upsert_doc(self, doc_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.records = [r for r in self.records if r.doc_id != doc_id]
        embeds = [e for r, e in zip(self.records, self.matrix)] if self.matrix is not None else []
        for c, emb in zip(chunks, embeddings):
            self.records.append(StoredChunk(
                doc_id=doc_id, chunk_id=c.chunk_id, text=c.text,
                section_path=c.section_path, chunk_type=c.chunk_type,
                token_count=c.token_count,
            ))
            embeds.append(np.asarray(emb, dtype=np.float32))
        if embeds:
            self.matrix = np.stack(embeds)
        else:
            self.matrix = None

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[StoredChunk]:
        if self.matrix is None or len(self.records) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        scores = self.matrix @ q  # embeddings are L2-normalized -> dot = cosine
        k = min(top_k, len(scores))
        idx = np.argsort(-scores)[:k]
        out = []
        for i in idx:
            rec = self.records[int(i)]
            out.append(StoredChunk(
                doc_id=rec.doc_id, chunk_id=rec.chunk_id, text=rec.text,
                section_path=rec.section_path, chunk_type=rec.chunk_type,
                token_count=rec.token_count, score=float(scores[i]),
            ))
        return out

    def chunks_in_section(self, section_path: str) -> list[StoredChunk]:
        prefix = section_path + " / "
        return [
            r for r in self.records
            if r.section_path == section_path or r.section_path.startswith(prefix)
        ]

    def get_sections(self) -> list[str]:
        return sorted({r.section_path for r in self.records})
