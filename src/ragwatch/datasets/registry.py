"""Dataset registry for dataset-agnostic RAGWatch experiments.

Maps short dataset names (for example ``squad`` and ``hotpotqa``) to lightweight
adapters that produce a unified :class:`~ragwatch.core.schema.RAGDataset`.
Adapters are constructed lazily and Hugging Face datasets are only loaded when a
dataset is actually requested, so importing this module never triggers heavy
imports or downloads.

To add a new dataset:
    1. Write an adapter (subclass of ``BaseQAAdapter``) that returns a RAGDataset.
    2. Register it here as a ``DatasetRegistryEntry``.
    3. Run the generic scripts with ``RAGWATCH_DATASET=<name>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ragwatch.config.env import get_env
from ragwatch.core.schema import RAGDataset

# Centralized environment variable names for dataset selection/configuration.
ENV_DATASET = "RAGWATCH_DATASET"
ENV_DATASET_SPLIT = "RAGWATCH_DATASET_SPLIT"
ENV_DATASET_MAX_EXAMPLES = "RAGWATCH_DATASET_MAX_EXAMPLES"
ENV_DATASET_LOAD_MAX_EXAMPLES = "RAGWATCH_DATASET_LOAD_MAX_EXAMPLES"

# Registered dataset names.
DATASET_SQUAD = "squad"
DATASET_HOTPOTQA = "hotpotqa"

DEFAULT_DATASET = DATASET_SQUAD


@dataclass(frozen=True)
class DatasetRegistryEntry:
    """A single registered dataset and how to build its adapter."""

    name: str
    default_split: str
    description: str
    adapter_factory: Callable[[], Any]


def _build_squad_adapter() -> Any:
    from ragwatch.datasets.adapters.squad_adapter import SquadAdapter

    return SquadAdapter()


def _build_hotpotqa_adapter() -> Any:
    from ragwatch.datasets.adapters.hotpotqa_adapter import HotpotQAAdapter

    return HotpotQAAdapter()


_REGISTRY: dict[str, DatasetRegistryEntry] = {
    DATASET_SQUAD: DatasetRegistryEntry(
        name=DATASET_SQUAD,
        default_split="validation",
        description="SQuAD v1.1 reading-comprehension QA (rajpurkar/squad).",
        adapter_factory=_build_squad_adapter,
    ),
    DATASET_HOTPOTQA: DatasetRegistryEntry(
        name=DATASET_HOTPOTQA,
        default_split="validation",
        description="HotpotQA multi-hop QA (hotpotqa/hotpot_qa, fullwiki config).",
        adapter_factory=_build_hotpotqa_adapter,
    ),
}


def get_available_dataset_names() -> list[str]:
    """Return the registered dataset names in sorted order."""
    return sorted(_REGISTRY.keys())


def get_dataset_entry(dataset_name: str) -> DatasetRegistryEntry:
    """Return the registry entry for ``dataset_name``.

    Raises:
        ValueError: If the dataset name is not registered. The error message
            lists the available dataset names.
    """
    key = (dataset_name or "").strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(get_available_dataset_names())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}."
        )
    return _REGISTRY[key]


def load_registered_dataset(
    dataset_name: str,
    split: str | None = None,
    max_examples: int | None = None,
) -> RAGDataset:
    """Load a registered dataset into a unified RAGDataset.

    Uses the entry's default split when ``split`` is not provided.
    """
    entry = get_dataset_entry(dataset_name)
    adapter = entry.adapter_factory()
    resolved_split = split or entry.default_split
    return adapter.load(split=resolved_split, max_examples=max_examples)


def _int_env(name: str, default: int | None) -> int | None:
    """Read an integer environment variable, returning ``default`` when unset."""
    value = get_env(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def dataset_name_from_env(default_dataset: str = DEFAULT_DATASET) -> str:
    """Return the dataset name from ``RAGWATCH_DATASET`` (default ``squad``)."""
    return (
        get_env(ENV_DATASET, default=default_dataset) or default_dataset
    ).strip().lower()


def dataset_split_from_env() -> str | None:
    """Return the split from ``RAGWATCH_DATASET_SPLIT`` (None when unset)."""
    value = get_env(ENV_DATASET_SPLIT)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def dataset_max_examples_from_env(default: int | None = None) -> int | None:
    """Return the number of QA examples to run (``RAGWATCH_DATASET_MAX_EXAMPLES``)."""
    return _int_env(ENV_DATASET_MAX_EXAMPLES, default)


def dataset_load_max_examples_from_env(default: int | None = None) -> int | None:
    """Return the number of rows to load (``RAGWATCH_DATASET_LOAD_MAX_EXAMPLES``)."""
    return _int_env(ENV_DATASET_LOAD_MAX_EXAMPLES, default)


def load_dataset_from_env(
    default_dataset: str = DEFAULT_DATASET,
    default_max_examples: int | None = None,
) -> RAGDataset:
    """Load a dataset selected and configured entirely from the environment.

    Reads ``RAGWATCH_DATASET`` (default ``squad``), ``RAGWATCH_DATASET_SPLIT``
    (default from the registry entry), and ``RAGWATCH_DATASET_MAX_EXAMPLES``
    (default ``default_max_examples``).
    """
    dataset_name = dataset_name_from_env(default_dataset)
    split = dataset_split_from_env()
    max_examples = dataset_max_examples_from_env(default_max_examples)
    return load_registered_dataset(
        dataset_name, split=split, max_examples=max_examples
    )
