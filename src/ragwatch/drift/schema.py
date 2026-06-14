"""Schemas and shared constants for the rule-based drift framework.

Terminology:
    Perturbation    = the actual rule-based modification applied to data.
    Drift scenario  = an experimental condition built from one or more
                      perturbations.
    Drift experiment = running RAGWatch on clean vs perturbed data and
                      comparing KPIs.

This module is the single source of truth for the controlled vocabulary of
perturbation types and their production interpretations, so those strings are
not redefined ad hoc across transforms and scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragwatch.core.schema import RAGDataset

# ---------------------------------------------------------------------------
# Controlled vocabulary of perturbation types (production-relevant names)
# ---------------------------------------------------------------------------

PERTURBATION_CLEAN = "clean"
PERTURBATION_CORPUS_CONTAMINATION = "corpus_contamination"
PERTURBATION_SOURCE_OUTAGE = "source_outage"
PERTURBATION_PARSER_NOISE = "parser_noise"
PERTURBATION_QUERY_DISTRIBUTION_SHIFT = "query_distribution_shift"

PERTURBATION_TYPES: tuple[str, ...] = (
    PERTURBATION_CLEAN,
    PERTURBATION_CORPUS_CONTAMINATION,
    PERTURBATION_SOURCE_OUTAGE,
    PERTURBATION_PARSER_NOISE,
    PERTURBATION_QUERY_DISTRIBUTION_SHIFT,
)

# Maps each perturbation type to a short, production-relevant interpretation
# string. Stored in scenario/dataset metadata so downstream analysis can
# explain *why* a degradation matters operationally.
PRODUCTION_INTERPRETATIONS: dict[str, str] = {
    PERTURBATION_CLEAN: "no_perturbation_baseline",
    PERTURBATION_CORPUS_CONTAMINATION: (
        "hard_negative_or_stale_content_entered_index"
    ),
    PERTURBATION_SOURCE_OUTAGE: (
        "connector_acl_or_ingestion_dropped_part_of_corpus"
    ),
    PERTURBATION_PARSER_NOISE: (
        "pdf_ocr_html_parsing_or_chunking_quality_degraded"
    ),
    PERTURBATION_QUERY_DISTRIBUTION_SHIFT: (
        "user_query_wording_and_phrasing_shifted"
    ),
}


@dataclass(frozen=True)
class DriftScenarioConfig:
    """Configuration describing a single drift scenario."""

    scenario_name: str
    perturbation_type: str
    severity: float
    random_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftedDataset:
    """A drifted dataset bundled with the config that produced it."""

    original_name: str
    scenario_name: str
    dataset: RAGDataset
    config: DriftScenarioConfig
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftTransformResult:
    """Result of applying a single perturbation transform.

    Optional convenience wrapper returning a perturbed dataset together with
    transform-specific metadata (e.g. removed doc IDs, injected count).
    """

    dataset: RAGDataset
    metadata: dict[str, Any] = field(default_factory=dict)
