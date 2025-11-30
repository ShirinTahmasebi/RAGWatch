"""FAISS-based retriever utilities for the HotpotQA pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ragwatch.utils import EnvKeys, env_path


DEFAULT_TOP_K = 5


def _resolve_index_dir(dataset_name: str, index_dir: str | Path | None) -> Path:
    base_dir = Path(index_dir) if index_dir is not None else env_path(EnvKeys.INDEX_DIR)
    return base_dir / dataset_name


def _ensure_index_dir(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)


def _index_files_exist(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def _to_documents(docs: List[Dict[str, str]]) -> List[Document]:
    return [
        Document(page_content=doc["text"], metadata={k: v for k, v in doc.items() if k != "text"})
        for doc in docs
    ]


def load_vectorstore(
    *,
    dataset_name: str,
    index_dir: str | Path | None = None,
    embeddings: Optional[Embeddings] = None,
):
    """Load the persisted FAISS index and return the underlying vectorstore."""

    index_dir = _resolve_index_dir(dataset_name, index_dir)
    if not _index_files_exist(index_dir):
        raise FileNotFoundError(f"FAISS index missing at: {index_dir}")

    embeddings = embeddings or OpenAIEmbeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_retriever(
    dataset_name: str,
    docs: List[Dict[str, str]],
    *,
    index_dir: str | Path | None = None,
    embeddings: Optional[Embeddings] = None,
    k: int = DEFAULT_TOP_K,
):
    """Build a FAISS index from raw docs, persist it, and return a retriever."""

    if not docs:
        raise ValueError("Cannot build retriever without documents")

    index_dir = _resolve_index_dir(dataset_name, index_dir)
    _ensure_index_dir(index_dir)

    embeddings = embeddings or OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(_to_documents(docs), embeddings)
    vectorstore.save_local(str(index_dir))
    return vectorstore.as_retriever(search_kwargs={"k": k})


def load_retriever(
    *,
    dataset_name: str,
    index_dir: str | Path | None = None,
    embeddings: Optional[Embeddings] = None,
    k: int = DEFAULT_TOP_K,
):
    """Load an existing FAISS index and return a retriever."""

    vectorstore = load_vectorstore(dataset_name=dataset_name, index_dir=index_dir, embeddings=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


