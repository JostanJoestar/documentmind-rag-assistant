from __future__ import annotations

import re

import pymupdf

from .models import PageText


class PDFExtractionError(ValueError):
    """Raised when a PDF cannot be opened or has no extractable text."""


def _clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes, document_name: str) -> list[PageText]:
    """Extract page text while retaining the original page number."""

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError(f"{document_name} is not a readable PDF.") from exc

    pages: list[PageText] = []
    try:
        for page_index, page in enumerate(document):
            text = _clean_text(page.get_text("text"))
            if text:
                pages.append(
                    PageText(
                        document_name=document_name,
                        page_number=page_index + 1,
                        text=text,
                    )
                )
    finally:
        document.close()

    if not pages:
        raise PDFExtractionError(
            f"{document_name} contains no extractable text. "
            "Scanned PDFs need OCR, which is not part of this MVP yet."
        )
    return pages
