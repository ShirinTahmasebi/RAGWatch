"""Tests for the optional pgvector inclusion in the retriever comparison example.

These tests exercise only the example's ``try_create_pgvector_spec`` helper
behavior when Postgres is not configured/available. They never require real
Docker, Postgres, or pgvector.
"""

import importlib.util
from pathlib import Path

import pytest

from ragwatch.core.schema import Document

EXAMPLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "run_squad_retriever_comparison.py"
)


def _load_example_module():
    """Import the example script as a module by file path."""
    spec = importlib.util.spec_from_file_location(
        "_squad_retriever_comparison_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example_module():
    return _load_example_module()


class TestTryCreatePgvectorSpec:
    def test_returns_none_when_url_missing(
        self, example_module, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("RAGWATCH_PGVECTOR_URL", raising=False)
        # Avoid a real .env overriding the test environment.
        monkeypatch.setattr(example_module, "load_env", lambda: None)

        spec = example_module.try_create_pgvector_spec(
            corpus=[], embedding_model=object(), generator=None
        )
        assert spec is None
        out = capsys.readouterr().out
        assert "Skipping pgvector" in out
        assert "docker compose up -d postgres" in out

    def test_returns_none_when_embedding_model_missing(
        self, example_module, monkeypatch, capsys
    ) -> None:
        monkeypatch.setenv("RAGWATCH_PGVECTOR_URL", "postgresql://x/y")

        spec = example_module.try_create_pgvector_spec(
            corpus=[], embedding_model=None, generator=None
        )
        assert spec is None
        assert "Skipping pgvector" in capsys.readouterr().out

    def test_returns_none_when_connection_fails(
        self, example_module, monkeypatch, capsys
    ) -> None:
        monkeypatch.setenv(
            "RAGWATCH_PGVECTOR_URL", "postgresql://invalid:invalid@127.0.0.1:1/none"
        )

        corpus = [Document(doc_id="d1", text="some text", metadata={})]
        spec = example_module.try_create_pgvector_spec(
            corpus=corpus, embedding_model=object(), generator=None
        )
        assert spec is None
        assert "Skipping pgvector" in capsys.readouterr().out
