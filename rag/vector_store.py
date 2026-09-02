from __future__ import annotations

import numpy as np

from .embeddings import Embedder
from .models import Chunk, SearchResult


class VectorIndex:
    """Minimal in-memory vector store using cosine similarity."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding matrix must contain one row per chunk")

        self.chunks = chunks
        self.embeddings = self._normalize_rows(embeddings.astype(np.float32))

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorIndex":
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        return cls(chunks, embeddings)

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def search(self, query: str, embedder: Embedder, top_k: int = 4) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Question must not be empty")

        query_vector = np.asarray(embedder.embed_query(query), dtype=np.float32).reshape(-1)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query and document embeddings have different dimensions")

        scores = self.embeddings @ query_vector
        count = min(max(top_k, 1), len(self.chunks))
        best_indices = np.argsort(scores)[::-1][:count]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in best_indices
        ]

