import unittest

import pymupdf
import numpy as np

from rag.models import SearchResult
from rag.pipeline import RAGPipeline


class FixedEmbedder:
    def embed_documents(self, texts):
        return np.array([[1.0, float(i + 1)] for i, _ in enumerate(texts)])

    def embed_query(self, text):
        del text
        return np.array([1.0, 1.0])


class RecordingGenerator:
    def __init__(self):
        self.sources: list[SearchResult] = []

    def answer(self, question, sources):
        self.sources = sources
        return f"Answer to: {question}"


def make_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Artificial intelligence supports document search. " * 12)
    content = document.tobytes()
    document.close()
    return content


class PipelineTests(unittest.TestCase):
    def test_ingest_and_ask_end_to_end(self):
        generator = RecordingGenerator()
        pipeline = RAGPipeline(FixedEmbedder(), generator)

        count = pipeline.ingest([("ai.pdf", make_pdf())], chunk_size=30, overlap=5)
        result = pipeline.ask("What does AI support?", top_k=2)

        self.assertGreater(count, 0)
        self.assertEqual(result.answer, "Answer to: What does AI support?")
        self.assertTrue(generator.sources)
        self.assertEqual(generator.sources[0].chunk.document_name, "ai.pdf")


if __name__ == "__main__":
    unittest.main()
