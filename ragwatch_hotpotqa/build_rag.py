"""LangChain-based RAG pipeline for the HotpotQA task."""
from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .retriever import load_retriever


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no supporting documents retrieved)"
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever=None):
    """Construct a simple question-answering chain backed by FAISS retriever."""

    retriever = retriever or load_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You answer questions using the provided context. Be concise."),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    rag_chain = (
        {"question": RunnablePassthrough(), "docs": retriever}
        | RunnableParallel(
            question=lambda x: x["question"],
            context=lambda x: _format_docs(x["docs"]),
        )
        | template
        | llm
    )

    return rag_chain
