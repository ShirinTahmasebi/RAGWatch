import tempfile
from pathlib import Path
import unittest

from langchain_core.embeddings import Embeddings

from ragwatch_client.vectorstores.faiss import FaissVectorStoreAdapter


class DummyEmbeddings(Embeddings):
    """Deterministic embedding model for tests (avoids API calls)."""

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    @staticmethod
    def _embed(text: str):
        base = float(len(text) % 7)
        return [base + i for i in range(4)]


def _sample_docs():
    return [
        {"doc_id": "d1", "text": "Alpha text", "title": "Alpha"},
        {"doc_id": "d2", "text": "Beta text", "title": "Beta"},
    ]


class HotpotQARetrieverTest(unittest.TestCase):
    def test_build_vectorstore_creates_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissVectorStoreAdapter(index_base_dir=tmpdir, embeddings=DummyEmbeddings())
            vectorstore = adapter.build("hotpotqa", _sample_docs())
            self.assertIsNotNone(vectorstore)

            index_dir = Path(tmpdir) / "hotpotqa"
            self.assertTrue((index_dir / "index.faiss").exists())
            self.assertTrue((index_dir / "index.pkl").exists())

    def test_load_fails_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissVectorStoreAdapter(index_base_dir=tmpdir, embeddings=DummyEmbeddings())
            with self.assertRaises(FileNotFoundError):
                adapter.load_if_exists("hotpotqa")


if __name__ == "__main__":
    unittest.main()
