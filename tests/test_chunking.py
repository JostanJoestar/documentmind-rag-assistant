import unittest

from rag.chunking import chunk_pages
from rag.models import PageText


class ChunkingTests(unittest.TestCase):
    def test_overlap_repeats_context(self):
        page = PageText("example.pdf", 2, " ".join(f"word{i}" for i in range(50)))

        chunks = chunk_pages([page], chunk_size=20, overlap=5)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text.split()[-5:], chunks[1].text.split()[:5])
        self.assertEqual(chunks[0].page_number, 2)
        self.assertEqual(chunks[0].document_name, "example.pdf")

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_pages([], chunk_size=20, overlap=20)


if __name__ == "__main__":
    unittest.main()

