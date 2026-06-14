"""Run a small drift experiment on a registered QA dataset (TF-IDF retriever).

This is the dataset-agnostic research entry point. Choose the dataset with the
``RAGWATCH_DATASET`` environment variable (default ``squad``):

    RAGWATCH_DATASET=squad    python examples/run_drift_experiment.py
    RAGWATCH_DATASET=hotpotqa python examples/run_drift_experiment.py

It runs RAGWatch over clean vs rule-based perturbed data and exports KPIs per
scenario, plus a combined drift_summary.csv with one row per scenario.

Dataset configuration (all optional):
    RAGWATCH_DATASET                    dataset name from the registry (default squad)
    RAGWATCH_DATASET_SPLIT              split name (default from the registry)
    RAGWATCH_DATASET_LOAD_MAX_EXAMPLES  rows to load to build the corpus (default 200)
    RAGWATCH_DATASET_MAX_EXAMPLES       QA examples run per scenario (default 20)

Semantic KPIs are computed by default using the local sentence-transformers
provider, and their averages are added to drift_summary.csv. The output
directory is dataset- and mode-aware:
    outputs/drift/squad_tfidf_semantic_local/        (default)
    outputs/drift/hotpotqa_tfidf_semantic_local/
    outputs/drift/squad_tfidf_semantic_openai/       (RAGWATCH_SEMANTIC_PROVIDER=openai)
    outputs/drift/squad_tfidf_base/                  (opt-out)

Each scenario directory contains a per-scenario experiment export and a
drifted_dataset/ snapshot (corpus.jsonl, qa_examples.jsonl, manifest.json).

Set RAGWATCH_SEMANTIC_PROVIDER to switch provider (local | openai | azure_openai).
Set RAGWATCH_DISABLE_SEMANTIC_KPIS=true to opt out and run deterministic-only.
Set RAGWATCH_DRIFT_OUTPUT_DIR to override the output directory verbatim.

A larger dataset slice is loaded to build a multi-document corpus (so
corpus-level perturbations are non-degenerate), while the number of QA examples
run per scenario stays small.
"""

import csv
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.config.env import get_env, load_env
from ragwatch.core.schema import RAGDataset
from ragwatch.datasets.registry import (
    dataset_load_max_examples_from_env,
    dataset_max_examples_from_env,
    dataset_name_from_env,
    dataset_split_from_env,
    get_dataset_entry,
    load_registered_dataset,
)
from ragwatch.drift.scenarios import create_drift_scenarios
from ragwatch.drift.persistence import export_drifted_dataset
from ragwatch.drift.schema import DriftedDataset
from ragwatch.experiments.exporters import export_experiment
from ragwatch.experiments.output_naming import drift_dir_name, drift_output_dir
from ragwatch.experiments.runner import ExperimentRunner
from ragwatch.experiments.schema import ExperimentConfig, ExperimentResult
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.report import KPIEngine
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever
from ragwatch.semantic.providers import (
    SemanticProviderConfigError,
    create_semantic_embedding_provider_from_env,
    resolve_semantic_provider_name,
    semantic_kpis_disabled,
    semantic_kpis_enabled,
)

# Load .env so dataset config and an optional output override are available.
load_env()

DEFAULT_DATASET = "squad"

# A larger slice is loaded to build a multi-document corpus (so corpus-level
# perturbations are non-degenerate), then the number of QA examples actually run
# per scenario is capped small.
DEFAULT_LOAD_MAX_EXAMPLES = 200
DEFAULT_MAX_EXAMPLES = 20
TOP_K = 3

# Optional override for the drift output directory. When set, it is used
# verbatim; otherwise the dataset-/mode-aware directory is used.
ENV_DRIFT_OUTPUT_DIR = "RAGWATCH_DRIFT_OUTPUT_DIR"

# Base drift summary columns and the KPIs they average (referencing the catalog
# so KPI names are not hardcoded as raw strings).
BASE_SUMMARY_AVG_COLUMNS: dict[str, KPIId] = {
    "avg_total_latency_ms": KPIId.TOTAL_LATENCY_MS,
    "avg_retrieval_score_mean": KPIId.RETRIEVAL_SCORE_MEAN,
    "avg_retrieval_redundancy": KPIId.RETRIEVAL_REDUNDANCY,
    "avg_answer_length_words": KPIId.ANSWER_LENGTH_WORDS,
}

# Semantic average columns, only added when semantic KPIs are enabled.
SEMANTIC_SUMMARY_AVG_COLUMNS: dict[str, KPIId] = {
    "avg_query_context_similarity_mean": KPIId.QUERY_CONTEXT_SIMILARITY_MEAN,
    "avg_answer_context_similarity_mean": KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN,
    "avg_answer_query_similarity": KPIId.ANSWER_QUERY_SIMILARITY,
}

SEMANTIC_FAILURE_MESSAGE = (
    "Semantic KPIs are enabled by default for this experiment.\n"
    "Install semantic dependencies with:\n"
    '  pip install -e ".[semantic]"\n\n'
    "For local semantic KPIs, use:\n"
    "  RAGWATCH_SEMANTIC_PROVIDER=local\n\n"
    "For OpenAI/Azure, configure the corresponding API variables in .env.\n\n"
    "To run without semantic KPIs, opt out explicitly:\n"
    "  RAGWATCH_DISABLE_SEMANTIC_KPIS=true \\\n"
    "      python examples/run_drift_experiment.py\n\n"
    "Underlying error: {error}"
)


def create_semantic_kpi_engine() -> SemanticKPIEngine:
    """Create a semantic KPI engine from env.

    Raises ``SemanticProviderConfigError`` if dependencies or configuration are
    missing. Callers are expected to fail clearly rather than silently dropping
    semantic KPIs for this research script.
    """
    provider = create_semantic_embedding_provider_from_env()
    return SemanticKPIEngine(provider=provider)


def summary_avg_columns(semantic_enabled: bool) -> dict[str, KPIId]:
    """Return the averaged-KPI columns, including semantic ones when enabled."""
    columns = dict(BASE_SUMMARY_AVG_COLUMNS)
    if semantic_enabled:
        columns.update(SEMANTIC_SUMMARY_AVG_COLUMNS)
    return columns


def drift_summary_columns(avg_columns: dict[str, KPIId]) -> list[str]:
    """Return the full drift_summary.csv column list for the given averages."""
    return [
        "scenario_name",
        "perturbation_type",
        "severity",
        "production_interpretation",
        "num_documents",
        "num_examples",
        "num_successful",
        "num_failed",
        *avg_columns.keys(),
    ]


def resolve_drift_output_dir(
    dataset_name: str,
    semantic_enabled: bool = False,
    semantic_provider_name: str | None = None,
) -> Path:
    """Resolve the drift output directory, honoring the env override.

    A ``RAGWATCH_DRIFT_OUTPUT_DIR`` override, when set, is used verbatim.
    Otherwise the directory is chosen from the dataset name and run mode so that
    datasets, semantic runs, and base runs never overwrite each other.
    """
    override = get_env(ENV_DRIFT_OUTPUT_DIR)
    if override:
        print(f"Using custom drift output directory: {override}")
        return Path(override)
    return drift_output_dir(
        dataset_name, semantic_enabled, semantic_provider_name
    )


def _average_kpis(
    result: ExperimentResult, kpi_ids: list[KPIId]
) -> dict[KPIId, float | None]:
    """Average the given KPIs over successful runs (None when unavailable)."""
    wanted = {str(kpi_id) for kpi_id in kpi_ids}
    sums: dict[str, float] = {name: 0.0 for name in wanted}
    counts: dict[str, int] = {name: 0 for name in wanted}

    for ex in result.examples:
        if not ex.succeeded or ex.kpi_report is None:
            continue
        for kpi in ex.kpi_report.results:
            if kpi.name in wanted and isinstance(kpi.value, (int, float)) and not (
                isinstance(kpi.value, bool)
            ):
                sums[kpi.name] += float(kpi.value)
                counts[kpi.name] += 1

    return {
        kpi_id: (sums[str(kpi_id)] / counts[str(kpi_id)])
        if counts[str(kpi_id)] > 0
        else None
        for kpi_id in kpi_ids
    }


def run_scenario(
    dataset_name: str,
    drifted: DriftedDataset,
    semantic_engine: SemanticKPIEngine | None = None,
    top_k: int = TOP_K,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> ExperimentResult:
    """Index a TF-IDF retriever on the scenario corpus and run the experiment."""
    dataset = drifted.dataset
    retriever = TfidfRetriever()
    retriever.index(dataset.corpus)
    client = BasicRAGClient(
        retriever=retriever, generator=HeuristicGenerator()
    )

    config = ExperimentConfig(
        experiment_name=f"{dataset_name}_drift_{drifted.scenario_name}",
        dataset_name=f"{dataset_name}_drift_{drifted.scenario_name}",
        retriever_name="tfidf",
        generator_name="heuristic",
        top_k=top_k,
        max_examples=max_examples,
        metadata={
            "scenario_name": drifted.scenario_name,
            "perturbation_type": drifted.config.perturbation_type,
            "severity": drifted.config.severity,
        },
    )
    runner = ExperimentRunner(
        client=client,
        kpi_engine=KPIEngine(),
        semantic_kpi_engine=semantic_engine,
    )
    return runner.run(
        qa_examples=dataset.qa_examples,
        config=config,
        compute_semantic_kpis=semantic_engine is not None,
    )


def build_drift_summary_row(
    drifted: DriftedDataset,
    result: ExperimentResult,
    avg_columns: dict[str, KPIId] | None = None,
) -> dict[str, Any]:
    """Build one drift_summary.csv row from a scenario and its result."""
    if avg_columns is None:
        avg_columns = BASE_SUMMARY_AVG_COLUMNS
    averages = _average_kpis(result, list(avg_columns.values()))
    row: dict[str, Any] = {
        "scenario_name": drifted.scenario_name,
        "perturbation_type": drifted.config.perturbation_type,
        "severity": drifted.config.severity,
        "production_interpretation": drifted.metadata.get(
            "production_interpretation"
        ),
        "num_documents": len(drifted.dataset.corpus),
        "num_examples": result.num_examples,
        "num_successful": result.num_successful,
        "num_failed": result.num_failed,
    }
    for column, kpi_id in avg_columns.items():
        row[column] = averages.get(kpi_id)
    return row


def write_drift_summary_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Write the combined drift summary CSV (one row per scenario)."""
    if fieldnames is None:
        fieldnames = drift_summary_columns(BASE_SUMMARY_AVG_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(default_dataset: str = DEFAULT_DATASET) -> None:
    # --- Resolve dataset selection/configuration from the environment ---
    dataset_name = dataset_name_from_env(default_dataset)
    entry = get_dataset_entry(dataset_name)
    split = dataset_split_from_env() or entry.default_split
    load_max = dataset_load_max_examples_from_env(DEFAULT_LOAD_MAX_EXAMPLES)
    run_max = dataset_max_examples_from_env(DEFAULT_MAX_EXAMPLES)

    print(
        f"Loading dataset '{dataset_name}' "
        f"(split={split}, max {load_max} rows)..."
    )
    loaded: RAGDataset = load_registered_dataset(
        dataset_name, split=split, max_examples=load_max
    )

    # Keep the full multi-document corpus but cap the QA examples actually run
    # per scenario, so perturbation ratios apply to exactly what is evaluated.
    dataset = RAGDataset(
        name=loaded.name,
        corpus=loaded.corpus,
        qa_examples=loaded.qa_examples[:run_max],
        metadata=dict(loaded.metadata),
    )
    print(f"Corpus: {len(dataset.corpus)} documents")
    print(
        f"QA examples: loaded {len(loaded.qa_examples)}, "
        f"running {len(dataset.qa_examples)} per scenario\n"
    )

    scenarios = create_drift_scenarios(dataset)
    print(f"Drift scenarios: {[s.scenario_name for s in scenarios]}\n")

    # --- Semantic KPIs are computed by default (opt out explicitly) ---
    semantic_enabled = semantic_kpis_enabled()
    provider_name: str | None = None
    semantic_engine: SemanticKPIEngine | None = None
    if semantic_enabled:
        try:
            semantic_engine = create_semantic_kpi_engine()
        except SemanticProviderConfigError as exc:
            print(SEMANTIC_FAILURE_MESSAGE.format(error=exc))
            raise SystemExit(1)
        provider_name = resolve_semantic_provider_name()
        print(f"Semantic KPIs enabled (provider: {provider_name}).")
    else:
        print("Semantic KPIs disabled via RAGWATCH_DISABLE_SEMANTIC_KPIS.")

    avg_columns = summary_avg_columns(semantic_enabled)
    output_dir = resolve_drift_output_dir(
        dataset_name, semantic_enabled, provider_name
    )
    summary_rows: list[dict[str, Any]] = []

    for drifted in scenarios:
        print(
            f"Running scenario '{drifted.scenario_name}' "
            f"({len(drifted.dataset.corpus)} docs)..."
        )
        scenario_output_dir = output_dir / drifted.scenario_name
        result = run_scenario(
            dataset_name,
            drifted,
            semantic_engine=semantic_engine,
            top_k=TOP_K,
            max_examples=run_max,
        )
        export_experiment(result, scenario_output_dir)
        export_drifted_dataset(drifted, scenario_output_dir / "drifted_dataset")
        summary_rows.append(
            build_drift_summary_row(drifted, result, avg_columns)
        )

    summary_path = output_dir / "drift_summary.csv"
    write_drift_summary_csv(
        summary_rows, summary_path, drift_summary_columns(avg_columns)
    )

    mode = drift_dir_name(dataset_name, semantic_enabled, provider_name)

    print("\n" + "=" * 60)
    print("Drift Experiment Summary")
    print("=" * 60)
    print(f"  Dataset:      {dataset_name} (split: {split})")
    print(f"  Mode:         {mode}")
    print(f"  Scenarios:    {len(scenarios)}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Summary CSV:  {summary_path}")
    print(
        "  Dataset snapshots: <scenario>/drifted_dataset/ "
        "(corpus.jsonl, qa_examples.jsonl, manifest.json)"
    )
    print()
    for row in summary_rows:
        latency = row["avg_total_latency_ms"]
        latency_str = f"{latency:.2f}" if latency is not None else "n/a"
        print(
            f"  {row['scenario_name']:<26} "
            f"docs={row['num_documents']:<5} "
            f"ok={row['num_successful']:<3} "
            f"avg_latency_ms={latency_str}"
        )
    print()


if __name__ == "__main__":
    main()
