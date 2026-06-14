"""Mode-aware output directory naming for RAGWatch experiments.

Centralizes the dataset- and semantic-mode-aware directory naming convention so
the experiment scripts, plot scripts, and the dashboard all agree on output
paths. This module is intentionally dependency-light (only ``pathlib``) so it
can be imported anywhere without pulling in heavy optional dependencies.

Examples of resolved directory names::

    squad_retrievers_semantic_local      hotpotqa_retrievers_semantic_local
    squad_retrievers_base                hotpotqa_retrievers_base
    squad_tfidf_semantic_local           hotpotqa_tfidf_semantic_local
    squad_tfidf_base                     hotpotqa_tfidf_base
"""

from __future__ import annotations

from pathlib import Path

COMPARISONS_ROOT = Path("outputs/comparisons")
DRIFT_ROOT = Path("outputs/drift")

# The experiment "kind" segment of a directory name.
COMPARISON_KIND = "retrievers"
DRIFT_KIND = "tfidf"

# The semantic-mode segment of a directory name.
SEMANTIC_MODE_PREFIX = "semantic_"
BASE_MODE = "base"


def semantic_mode_label(semantic_enabled: bool, provider_name: str | None) -> str:
    """Return the mode segment of a directory name.

    Semantic runs use ``semantic_<provider>`` (for example ``semantic_local``);
    opt-out runs use ``base``.
    """
    if semantic_enabled and provider_name:
        return f"{SEMANTIC_MODE_PREFIX}{provider_name}"
    return BASE_MODE


def comparison_dir_name(
    dataset_name: str, semantic_enabled: bool, provider_name: str | None
) -> str:
    """Return the mode-aware retriever comparison directory name."""
    mode = semantic_mode_label(semantic_enabled, provider_name)
    return f"{dataset_name}_{COMPARISON_KIND}_{mode}"


def comparison_output_dir(
    dataset_name: str, semantic_enabled: bool, provider_name: str | None
) -> Path:
    """Return the full comparison output directory under ``COMPARISONS_ROOT``."""
    return COMPARISONS_ROOT / comparison_dir_name(
        dataset_name, semantic_enabled, provider_name
    )


def drift_dir_name(
    dataset_name: str, semantic_enabled: bool, provider_name: str | None
) -> str:
    """Return the mode-aware drift experiment directory name."""
    mode = semantic_mode_label(semantic_enabled, provider_name)
    return f"{dataset_name}_{DRIFT_KIND}_{mode}"


def drift_output_dir(
    dataset_name: str, semantic_enabled: bool, provider_name: str | None
) -> Path:
    """Return the full drift output directory under ``DRIFT_ROOT``."""
    return DRIFT_ROOT / drift_dir_name(dataset_name, semantic_enabled, provider_name)
