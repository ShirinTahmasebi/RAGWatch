import tempfile
from pathlib import Path
import unittest

from langchain_core.embeddings import Embeddings

from ragwatch_client.datasets.hotpotqa.retriever import build_retriever, load_retriever


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
    def test_build_retriever_creates_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = DummyEmbeddings()
            build_retriever("hotpotqa", _sample_docs(), index_dir=tmpdir, embeddings=embeddings, k=2)

            index_dir = Path(tmpdir)
            self.assertTrue((index_dir / "hotpotqa" / "index.faiss").exists())
            self.assertTrue((index_dir / "hotpotqa" / "index.pkl").exists())

    def test_load_retriever_fails_without_index(self) -> None:
        embeddings = DummyEmbeddings()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_retriever(dataset_name="hotpotqa", index_dir=tmpdir, embeddings=embeddings, k=2)


if __name__ == "__main__":
    unittest.main()
