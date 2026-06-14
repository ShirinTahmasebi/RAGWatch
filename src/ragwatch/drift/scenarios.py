"""Drift scenario builders.

A *drift scenario* is an experimental condition created from one or more
perturbations. These builders wrap the deterministic transforms in
``ragwatch.drift.transforms`` and attach a :class:`DriftScenarioConfig` so each
condition is reproducible and self-describing.
"""

from __future__ import annotations

from ragwatch.core.schema import RAGDataset
from ragwatch.drift.schema import (
    PERTURBATION_CLEAN,
    PERTURBATION_CORPUS_CONTAMINATION,
    PERTURBATION_PARSER_NOISE,
    PERTURBATION_QUERY_DISTRIBUTION_SHIFT,
    PERTURBATION_SOURCE_OUTAGE,
    PRODUCTION_INTERPRETATIONS,
    DriftedDataset,
    DriftScenarioConfig,
)
from ragwatch.drift.transforms import (
    apply_parser_noise,
    apply_query_distribution_shift,
    copy_dataset,
    inject_corpus_contamination,
    simulate_source_outage,
)


def create_drift_scenario(
    dataset: RAGDataset,
    config: DriftScenarioConfig,
) -> DriftedDataset:
    """Build a single drifted dataset from a scenario config.

    The input dataset is never mutated. Raises ``ValueError`` for unknown
    perturbation types.
    """
    perturbation = config.perturbation_type
    seed = config.random_seed
    severity = config.severity

    if perturbation == PERTURBATION_CLEAN:
        drifted = copy_dataset(dataset)
    elif perturbation == PERTURBATION_CORPUS_CONTAMINATION:
        drifted = inject_corpus_contamination(dataset, severity, seed=seed)
    elif perturbation == PERTURBATION_SOURCE_OUTAGE:
        drifted = simulate_source_outage(dataset, severity, seed=seed)
    elif perturbation == PERTURBATION_PARSER_NOISE:
        drifted = apply_parser_noise(dataset, severity, seed=seed)
    elif perturbation == PERTURBATION_QUERY_DISTRIBUTION_SHIFT:
        drifted = apply_query_distribution_shift(dataset, severity, seed=seed)
    else:
        raise ValueError(
            f"Unknown perturbation_type: {perturbation!r}. "
            f"Expected one of: clean, corpus_contamination, source_outage, "
            f"parser_noise, query_distribution_shift."
        )

    return DriftedDataset(
        original_name=dataset.name,
        scenario_name=config.scenario_name,
        dataset=drifted,
        config=config,
        metadata={
            "perturbation_type": perturbation,
            "severity": severity,
            "random_seed": seed,
            "production_interpretation": PRODUCTION_INTERPRETATIONS[perturbation],
            "num_documents": len(drifted.corpus),
            "num_examples": len(drifted.qa_examples),
            **dict(config.metadata),
        },
    )


# Default scenario matrix. Each entry is (scenario_name, perturbation_type,
# severity). Severities are expressed as ratios in [0.0, 1.0].
_DEFAULT_SCENARIO_MATRIX: tuple[tuple[str, str, float], ...] = (
    ("clean", PERTURBATION_CLEAN, 0.0),
    ("corpus_contamination_10", PERTURBATION_CORPUS_CONTAMINATION, 0.1),
    ("corpus_contamination_30", PERTURBATION_CORPUS_CONTAMINATION, 0.3),
    ("source_outage_10", PERTURBATION_SOURCE_OUTAGE, 0.1),
    ("source_outage_30", PERTURBATION_SOURCE_OUTAGE, 0.3),
    ("parser_noise_10", PERTURBATION_PARSER_NOISE, 0.1),
    ("query_shift_10", PERTURBATION_QUERY_DISTRIBUTION_SHIFT, 0.1),
    ("query_shift_30", PERTURBATION_QUERY_DISTRIBUTION_SHIFT, 0.3),
)


def create_drift_scenarios(
    dataset: RAGDataset,
    seed: int = 42,
) -> list[DriftedDataset]:
    """Build the default set of drift scenarios for a dataset.

    Returns one :class:`DriftedDataset` per scenario, starting with ``clean``
    (a copied, unperturbed dataset) followed by contamination, source outage,
    parser noise, and query shift conditions.
    """
    scenarios: list[DriftedDataset] = []
    for scenario_name, perturbation_type, severity in _DEFAULT_SCENARIO_MATRIX:
        config = DriftScenarioConfig(
            scenario_name=scenario_name,
            perturbation_type=perturbation_type,
            severity=severity,
            random_seed=seed,
        )
        scenarios.append(create_drift_scenario(dataset, config))
    return scenarios
