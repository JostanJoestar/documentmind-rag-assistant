from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page."""

    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document with source metadata."""

    chunk_id: str
    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SearchResult]

