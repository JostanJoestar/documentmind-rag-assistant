from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local multilingual embeddings; no API key or document upload required."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is missing. Run: pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

