"""Tests for importable helpers in the drift experiment example.

Loads the generic drift example module via importlib and exercises its pure
helper functions with toy data, and confirms the SQuAD wrapper defaults to
SQuAD. Does not require Hugging Face, OpenAI, Azure, Postgres, Chroma, Qdrant,
Streamlit, Plotly, or API keys.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.drift.schema import DriftedDataset, DriftScenarioConfig
from ragwatch.experiments.schema import (
    ExperimentConfig,
    ExperimentExampleResult,
    ExperimentResult,
)
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.schema import KPIReport, KPIResult

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "run_drift_experiment.py"
SQUAD_WRAPPER_PATH = EXAMPLES_DIR / "run_squad_drift_experiment.py"


def _load_module(name: str, path: Path):
    # The wrapper imports the generic module by name, so the examples dir must be
    # importable.
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def drift_module():
    return _load_module("_drift_example", EXAMPLE_PATH)


@pytest.fixture(autouse=True)
def _clear_drift_env(monkeypatch):
    monkeypatch.delenv("RAGWATCH_DRIFT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("RAGWATCH_DISABLE_SEMANTIC_KPIS", raising=False)
    monkeypatch.delenv("RAGWATCH_SEMANTIC_PROVIDER", raising=False)
    monkeypatch.delenv("RAGWATCH_DATASET", raising=False)


def _drifted() -> DriftedDataset:
    dataset = RAGDataset(
        name="toy",
        corpus=[Document(doc_id="d0", text="text")],
        qa_examples=[QAExample(example_id="q0", question="q?", answers=["a"])],
    )
    config = DriftScenarioConfig(
        scenario_name="clean",
        perturbation_type="clean",
        severity=0.0,
        random_seed=42,
    )
    return DriftedDataset(
        original_name="toy",
        scenario_name="clean",
        dataset=dataset,
        config=config,
        metadata={"production_interpretation": "no_perturbation_baseline"},
    )


def _kpi(name: str, value: float) -> KPIResult:
    return KPIResult(
        name=name,
        value=value,
        category="runtime",
        stage="end_to_end",
        source="rag_run",
        description="",
    )


def _result_with_kpis() -> ExperimentResult:
    report = KPIReport(
        run_id="r1",
        query="q?",
        results=[
            _kpi(str(KPIId.TOTAL_LATENCY_MS), 10.0),
            _kpi(str(KPIId.RETRIEVAL_SCORE_MEAN), 0.5),
            _kpi(str(KPIId.RETRIEVAL_REDUNDANCY), 0.2),
            _kpi(str(KPIId.ANSWER_LENGTH_WORDS), 4.0),
        ],
    )
    example = ExperimentExampleResult(
        example_id="q0",
        question="q?",
        gold_answers=["a"],
        run=object(),  # truthy run so succeeded is True
        kpi_report=report,
    )
    config = ExperimentConfig(
        experiment_name="e",
        dataset_name="d",
        retriever_name="tfidf",
        generator_name="heuristic",
    )
    return ExperimentResult(config=config, examples=[example])


def _result_with_semantic_kpis() -> ExperimentResult:
    report = KPIReport(
        run_id="r1",
        query="q?",
        results=[
            _kpi(str(KPIId.TOTAL_LATENCY_MS), 10.0),
            _kpi(str(KPIId.RETRIEVAL_SCORE_MEAN), 0.5),
            _kpi(str(KPIId.RETRIEVAL_REDUNDANCY), 0.2),
            _kpi(str(KPIId.ANSWER_LENGTH_WORDS), 4.0),
            _kpi(str(KPIId.QUERY_CONTEXT_SIMILARITY_MEAN), 0.6),
            _kpi(str(KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN), 0.7),
            _kpi(str(KPIId.ANSWER_QUERY_SIMILARITY), 0.8),
        ],
    )
    example = ExperimentExampleResult(
        example_id="q0",
        question="q?",
        gold_answers=["a"],
        run=object(),  # truthy run so succeeded is True
        kpi_report=report,
    )
    config = ExperimentConfig(
        experiment_name="e",
        dataset_name="d",
        retriever_name="tfidf",
        generator_name="heuristic",
    )
    return ExperimentResult(config=config, examples=[example])


def test_resolve_drift_output_dir_default_is_semantic_local(drift_module) -> None:
    assert drift_module.semantic_kpis_enabled() is True
    assert drift_module.resolve_semantic_provider_name() == "local"
    out = drift_module.resolve_drift_output_dir(
        "squad", semantic_enabled=True, semantic_provider_name="local"
    )
    assert out == Path("outputs/drift/squad_tfidf_semantic_local")


def test_resolve_drift_output_dir_opt_out(drift_module, monkeypatch) -> None:
    monkeypatch.setenv("RAGWATCH_DISABLE_SEMANTIC_KPIS", "true")
    assert drift_module.semantic_kpis_enabled() is False
    out = drift_module.resolve_drift_output_dir(
        "squad", semantic_enabled=False, semantic_provider_name=None
    )
    assert out == Path("outputs/drift/squad_tfidf_base")


def test_resolve_drift_output_dir_hotpotqa(drift_module) -> None:
    out = drift_module.resolve_drift_output_dir(
        "hotpotqa", semantic_enabled=True, semantic_provider_name="local"
    )
    assert out == Path("outputs/drift/hotpotqa_tfidf_semantic_local")


def test_resolve_drift_output_dir_override(drift_module, monkeypatch) -> None:
    monkeypatch.setenv("RAGWATCH_DRIFT_OUTPUT_DIR", "outputs/drift/custom")
    assert drift_module.resolve_drift_output_dir("squad") == Path(
        "outputs/drift/custom"
    )


def test_squad_wrapper_defaults_to_squad(drift_module) -> None:
    wrapper = _load_module("_squad_drift_wrapper", SQUAD_WRAPPER_PATH)
    assert wrapper.DEFAULT_DATASET == "squad"
    assert wrapper.main.__module__ == "run_drift_experiment"


def test_build_drift_summary_row(drift_module) -> None:
    row = drift_module.build_drift_summary_row(_drifted(), _result_with_kpis())
    assert row["scenario_name"] == "clean"
    assert row["perturbation_type"] == "clean"
    assert row["num_documents"] == 1
    assert row["num_examples"] == 1
    assert row["num_successful"] == 1
    assert row["avg_total_latency_ms"] == 10.0
    assert row["avg_retrieval_score_mean"] == 0.5
    assert row["avg_retrieval_redundancy"] == 0.2
    assert row["avg_answer_length_words"] == 4.0


def test_build_drift_summary_row_with_semantic(drift_module) -> None:
    avg_columns = drift_module.summary_avg_columns(True)
    row = drift_module.build_drift_summary_row(
        _drifted(), _result_with_semantic_kpis(), avg_columns
    )
    assert row["avg_total_latency_ms"] == 10.0
    assert row["avg_query_context_similarity_mean"] == 0.6
    assert row["avg_answer_context_similarity_mean"] == 0.7
    assert row["avg_answer_query_similarity"] == 0.8


def test_write_drift_summary_csv(drift_module, tmp_path) -> None:
    row = drift_module.build_drift_summary_row(_drifted(), _result_with_kpis())
    out = tmp_path / "drift_summary.csv"
    drift_module.write_drift_summary_csv([row], out)
    content = out.read_text(encoding="utf-8")
    assert "scenario_name" in content.splitlines()[0]
    assert "clean" in content
