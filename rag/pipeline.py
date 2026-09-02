from __future__ import annotations

from collections.abc import Iterable

from .chunking import chunk_pages
from .embeddings import Embedder
from .generation import Generator
from .models import AnswerResult
from .pdf import extract_pages
from .vector_store import VectorIndex


class RAGPipeline:
    def __init__(self, embedder: Embedder, generator: Generator):
        self.embedder = embedder
        self.generator = generator
        self.index: VectorIndex | None = None

    def ingest(
        self,
        documents: Iterable[tuple[str, bytes]],
        chunk_size: int = 180,
        overlap: int = 35,
    ) -> int:
        pages = []
        for name, content in documents:
            pages.extend(extract_pages(content, name))

        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        self.index = VectorIndex.build(chunks, self.embedder)
        return len(chunks)

    def ask(self, question: str, top_k: int = 4) -> AnswerResult:
        if self.index is None:
            raise RuntimeError("Upload and process at least one document first.")

        sources = self.index.search(question, self.embedder, top_k=top_k)
        answer = self.generator.answer(question, sources)
        return AnswerResult(answer=answer, sources=sources)

