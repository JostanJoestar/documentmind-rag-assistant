from __future__ import annotations

import re

from .models import Chunk, PageText


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 180,
    overlap: int = 35,
) -> list[Chunk]:
    """Split pages into overlapping word windows.

    Chunk boundaries never cross pages. This makes page-level citations exact.
    """

    if chunk_size < 20:
        raise ValueError("chunk_size must be at least 20 words")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for page in pages:
        words = _words(page.text)
        for chunk_number, start in enumerate(range(0, len(words), step), start=1):
            window = words[start : start + chunk_size]
            if not window:
                continue
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{page.document_name}:p{page.page_number}:c{chunk_number}"
                    ),
                    document_name=page.document_name,
                    page_number=page.page_number,
                    text=" ".join(window),
                )
            )
            if start + chunk_size >= len(words):
                break

    return chunks

