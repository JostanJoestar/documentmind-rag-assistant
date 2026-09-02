import unittest

import numpy as np

from rag.models import Chunk
from rag.vector_store import VectorIndex


class FakeEmbedder:
    vectors = {
        "apples": np.array([1.0, 0.0]),
        "bananas": np.array([0.0, 1.0]),
        "fruit question": np.array([0.9, 0.1]),
    }

    def embed_documents(self, texts):
        return np.vstack([self.vectors[text] for text in texts])

    def embed_query(self, text):
        return self.vectors[text]


class VectorStoreTests(unittest.TestCase):
    def test_returns_most_similar_chunk_first(self):
        chunks = [
            Chunk("1", "food.pdf", 1, "apples"),
            Chunk("2", "food.pdf", 2, "bananas"),
        ]
        embedder = FakeEmbedder()
        index = VectorIndex.build(chunks, embedder)

        results = index.search("fruit question", embedder, top_k=2)

        self.assertEqual(results[0].chunk.chunk_id, "1")
        self.assertGreater(results[0].score, results[1].score)


if __name__ == "__main__":
    unittest.main()

