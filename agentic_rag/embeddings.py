"""Local, open-source dense embeddings via sentence-transformers.

Default model: all-MiniLM-L6-v2 (384-dim, ~90 MB, downloaded once from the
Hugging Face hub on first use). Embeddings are L2-normalized, so cosine
similarity between vectors is a plain dot product.
"""
from __future__ import annotations

import os

import numpy as np

# This pipeline is torch-only. Some machines have TensorFlow/Keras installed
# whose (broken) lazy imports would otherwise crash transformers' import
# chain - disable TF integration explicitly before any transformers import.
os.environ.setdefault("USE_TF", "0")


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        fn = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(fn())

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
