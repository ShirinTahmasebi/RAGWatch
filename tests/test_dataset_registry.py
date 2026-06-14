"""Tests for the dataset registry (datasets/registry.py).

These tests are deterministic and never touch the network or Hugging Face.
Loading is exercised by registering fake entries whose adapters return tiny
in-memory RAGDataset objects and record how they were called.
"""

from __future__ import annotations

import pytest

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.datasets import registry


@pytest.fixture(autouse=True)
def _clear_dataset_env(monkeypatch):
    """Ensure dataset env vars never leak between tests."""
    for name in (
        registry.ENV_DATASET,
        registry.ENV_DATASET_SPLIT,
        registry.ENV_DATASET_MAX_EXAMPLES,
        registry.ENV_DATASET_LOAD_MAX_EXAMPLES,
    ):
        monkeypatch.delenv(name, raising=False)
    yield


class _FakeAdapter:
    """A fake adapter that records its load() arguments."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []

    def load(self, split=None, max_examples=None) -> RAGDataset:
        self.calls.append({"split": split, "max_examples": max_examples})
        return RAGDataset(
            name=self.name,
            corpus=[Document(doc_id="d1", text="hello world")],
            qa_examples=[QAExample(example_id="q1", question="hi?", answers=["x"])],
            metadata={"split": split, "max_examples": max_examples},
        )


@pytest.fixture
def fake_registry(monkeypatch):
    """Replace the registry with two fake datasets and return their adapters."""
    squad_adapter = _FakeAdapter("fake_squad")
    other_adapter = _FakeAdapter("fake_other")

    fake = {
        "squad": registry.DatasetRegistryEntry(
            name="squad",
            default_split="validation",
            description="fake squad",
            adapter_factory=lambda: squad_adapter,
        ),
        "otherset": registry.DatasetRegistryEntry(
            name="otherset",
            default_split="train",
            description="fake other",
            adapter_factory=lambda: other_adapter,
        ),
    }
    monkeypatch.setattr(registry, "_REGISTRY", fake)
    return {"squad": squad_adapter, "otherset": other_adapter}


# --- Registry contents --------------------------------------------------------


def test_registry_includes_squad_and_hotpotqa():
    names = registry.get_available_dataset_names()
    assert "squad" in names
    assert "hotpotqa" in names


def test_get_available_dataset_names_is_sorted():
    names = registry.get_available_dataset_names()
    assert names == sorted(names)


def test_default_dataset_is_squad():
    assert registry.DEFAULT_DATASET == "squad"


# --- Entry lookup -------------------------------------------------------------


def test_get_dataset_entry_normalizes_name(fake_registry):
    entry = registry.get_dataset_entry("  SQUAD  ")
    assert entry.name == "squad"


def test_get_dataset_entry_unknown_raises_with_available_names():
    with pytest.raises(ValueError) as exc_info:
        registry.get_dataset_entry("does_not_exist")
    message = str(exc_info.value)
    assert "does_not_exist" in message
    # The message should help the user by listing valid names.
    assert "squad" in message
    assert "hotpotqa" in message


# --- Loading via the registry -------------------------------------------------


def test_load_registered_dataset_uses_default_split(fake_registry):
    dataset = registry.load_registered_dataset("squad")
    assert dataset.name == "fake_squad"
    assert fake_registry["squad"].calls == [
        {"split": "validation", "max_examples": None}
    ]


def test_load_registered_dataset_passes_split_and_max(fake_registry):
    registry.load_registered_dataset("otherset", split="dev", max_examples=7)
    assert fake_registry["otherset"].calls == [
        {"split": "dev", "max_examples": 7}
    ]


# --- Environment-driven selection ---------------------------------------------


def test_dataset_name_from_env_defaults_to_squad():
    assert registry.dataset_name_from_env() == "squad"


def test_dataset_name_from_env_respects_env(monkeypatch):
    monkeypatch.setenv(registry.ENV_DATASET, "HotpotQA")
    assert registry.dataset_name_from_env() == "hotpotqa"


def test_dataset_split_from_env(monkeypatch):
    assert registry.dataset_split_from_env() is None
    monkeypatch.setenv(registry.ENV_DATASET_SPLIT, "train")
    assert registry.dataset_split_from_env() == "train"


def test_dataset_max_examples_from_env(monkeypatch):
    assert registry.dataset_max_examples_from_env(default=11) == 11
    monkeypatch.setenv(registry.ENV_DATASET_MAX_EXAMPLES, "5")
    assert registry.dataset_max_examples_from_env(default=11) == 5


def test_dataset_load_max_examples_from_env(monkeypatch):
    assert registry.dataset_load_max_examples_from_env(default=200) == 200
    monkeypatch.setenv(registry.ENV_DATASET_LOAD_MAX_EXAMPLES, "42")
    assert registry.dataset_load_max_examples_from_env(default=200) == 42


def test_load_dataset_from_env_uses_env_selection(fake_registry, monkeypatch):
    monkeypatch.setenv(registry.ENV_DATASET, "otherset")
    monkeypatch.setenv(registry.ENV_DATASET_SPLIT, "dev")
    monkeypatch.setenv(registry.ENV_DATASET_MAX_EXAMPLES, "3")

    dataset = registry.load_dataset_from_env(default_max_examples=99)

    assert dataset.name == "fake_other"
    assert fake_registry["otherset"].calls == [
        {"split": "dev", "max_examples": 3}
    ]


def test_load_dataset_from_env_defaults(fake_registry):
    dataset = registry.load_dataset_from_env(default_max_examples=8)
    # Defaults to squad with its registry default split.
    assert dataset.name == "fake_squad"
    assert fake_registry["squad"].calls == [
        {"split": "validation", "max_examples": 8}
    ]
