"""FAISS-based retriever utilities for the HotpotQA pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .data import build_document_corpus
from ragwatch.settings import env_path

DEFAULT_TOP_K = 5


def _resolve_index_dir(index_dir: str | Path | None) -> Path:
    if index_dir is not None:
        return Path(index_dir)
    return env_path("RAGWATCH_HOTPOTQA_INDEX_DIR")


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
    index_dir: str | Path | None = None,
    embeddings: Optional[Embeddings] = None,
):
    """Load the persisted FAISS index and return the underlying vectorstore."""

    index_dir = _resolve_index_dir(index_dir)
    if not _index_files_exist(index_dir):
        raise FileNotFoundError(f"FAISS index missing at: {index_dir}")

    embeddings = embeddings or OpenAIEmbeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_retriever(
    docs: List[Dict[str, str]],
    *,
    index_dir: str | Path | None = None,
    embeddings: Optional[Embeddings] = None,
    k: int = DEFAULT_TOP_K,
):
    """Build a FAISS index from raw docs, persist it, and return a retriever."""

    if not docs:
        raise ValueError("Cannot build retriever without documents")

    index_dir = _resolve_index_dir(index_dir)
    _ensure_index_dir(index_dir)

    embeddings = embeddings or OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(_to_documents(docs), embeddings)
    vectorstore.save_local(str(index_dir))
    return vectorstore.as_retriever(search_kwargs={"k": k})


def load_retriever(
    *,
    index_dir: str | Path | None = None,
    embeddings: Optional[Embeddings] = None,
    k: int = DEFAULT_TOP_K,
):
    """Load an existing FAISS index and return a retriever."""

    vectorstore = load_vectorstore(index_dir=index_dir, embeddings=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def ensure_retriever(
    *,
    index_dir: str | Path | None = None,
    doc_builder: Callable[[], List[Dict[str, str]]] = build_document_corpus,
    embeddings: Optional[Embeddings] = None,
    k: int = DEFAULT_TOP_K,
    force_rebuild: bool = False,
):
    """Load an existing retriever or build one if the index is missing."""

    index_dir = _resolve_index_dir(index_dir)

    if force_rebuild or not _index_files_exist(index_dir):
        docs = doc_builder()
        if not docs:
            raise ValueError("Document builder returned no records; cannot create index")
        build_retriever(docs, index_dir=index_dir, embeddings=embeddings, k=k)

    return load_retriever(index_dir=index_dir, embeddings=embeddings, k=k)
