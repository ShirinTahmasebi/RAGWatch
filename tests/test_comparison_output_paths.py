"""Tests for dataset-/mode-aware comparison and figure output resolution.

These tests exercise the small helper functions in the generic comparison and
plot example scripts (and confirm the SQuAD wrappers default to SQuAD). They
never require Hugging Face, Chroma, Qdrant, pgvector, Postgres, OpenAI, Azure,
Streamlit, or Plotly.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
COMPARISON_EXAMPLE = EXAMPLES_DIR / "run_retriever_comparison.py"
PLOTS_EXAMPLE = EXAMPLES_DIR / "generate_comparison_plots.py"
SQUAD_COMPARISON_WRAPPER = EXAMPLES_DIR / "run_squad_retriever_comparison.py"


def _load_module(name: str, path: Path):
    # The examples import each other by module name, so the examples dir must be
    # importable while loading the wrapper modules.
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def comparison_module():
    return _load_module("_comparison_example", COMPARISON_EXAMPLE)


@pytest.fixture
def plots_module():
    return _load_module("_plots_example", PLOTS_EXAMPLE)


@pytest.fixture(autouse=True)
def _clear_output_env(monkeypatch):
    """Ensure output-dir, dataset, and semantic overrides are unset by default."""
    for name in (
        "RAGWATCH_COMPARISON_OUTPUT_DIR",
        "RAGWATCH_FIGURE_OUTPUT_DIR",
        "RAGWATCH_DISABLE_SEMANTIC_KPIS",
        "RAGWATCH_SEMANTIC_PROVIDER",
        "RAGWATCH_DATASET",
        "RAGWATCH_DATASET_SPLIT",
        "RAGWATCH_DATASET_MAX_EXAMPLES",
    ):
        monkeypatch.delenv(name, raising=False)


class TestComparisonOutputDir:
    def test_opt_out_resolves_to_base_dir(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            "squad",
            semantic_enabled=False,
            semantic_provider_name=None,
        )
        assert out == Path("outputs/comparisons/squad_retrievers_base")

    def test_semantic_local_resolves(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            "squad",
            semantic_enabled=True,
            semantic_provider_name="local",
        )
        assert out == Path("outputs/comparisons/squad_retrievers_semantic_local")

    def test_semantic_openai_resolves(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            "squad",
            semantic_enabled=True,
            semantic_provider_name="openai",
        )
        assert out == Path("outputs/comparisons/squad_retrievers_semantic_openai")

    def test_semantic_azure_resolves(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            "squad",
            semantic_enabled=True,
            semantic_provider_name="azure_openai",
        )
        assert out == Path(
            "outputs/comparisons/squad_retrievers_semantic_azure_openai"
        )

    def test_hotpotqa_dataset_in_path(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            "hotpotqa",
            semantic_enabled=True,
            semantic_provider_name="local",
        )
        assert out == Path(
            "outputs/comparisons/hotpotqa_retrievers_semantic_local"
        )

    def test_default_is_semantic_local(self, comparison_module) -> None:
        # With no env overrides, semantic is enabled and the provider is local.
        assert comparison_module.semantic_kpis_enabled() is True
        assert comparison_module.resolve_semantic_provider_name() == "local"
        out = comparison_module.resolve_comparison_output_dir(
            "squad",
            semantic_enabled=comparison_module.semantic_kpis_enabled(),
            semantic_provider_name=comparison_module.resolve_semantic_provider_name(),
        )
        assert out == Path("outputs/comparisons/squad_retrievers_semantic_local")

    def test_opt_out_flag_disables_semantic(
        self, comparison_module, monkeypatch
    ) -> None:
        monkeypatch.setenv("RAGWATCH_DISABLE_SEMANTIC_KPIS", "true")
        assert comparison_module.semantic_kpis_enabled() is False
        out = comparison_module.resolve_comparison_output_dir(
            "squad",
            semantic_enabled=comparison_module.semantic_kpis_enabled(),
            semantic_provider_name=None,
        )
        assert out == Path("outputs/comparisons/squad_retrievers_base")

    def test_override_wins(self, comparison_module, monkeypatch, capsys) -> None:
        monkeypatch.setenv(
            "RAGWATCH_COMPARISON_OUTPUT_DIR", "outputs/comparisons/my_custom_run"
        )
        out = comparison_module.resolve_comparison_output_dir(
            "squad",
            semantic_enabled=True,
            semantic_provider_name="local",
        )
        assert out == Path("outputs/comparisons/my_custom_run")
        assert "custom comparison output directory" in capsys.readouterr().out.lower()


class TestComparisonModeName:
    def test_base_name(self, comparison_module) -> None:
        assert (
            comparison_module.comparison_dir_name("squad", False, None)
            == "squad_retrievers_base"
        )

    def test_semantic_name(self, comparison_module) -> None:
        assert (
            comparison_module.comparison_dir_name("squad", True, "local")
            == "squad_retrievers_semantic_local"
        )

    def test_hotpotqa_name(self, comparison_module) -> None:
        assert (
            comparison_module.comparison_dir_name("hotpotqa", True, "local")
            == "hotpotqa_retrievers_semantic_local"
        )


class TestSquadWrapperDefaults:
    def test_wrapper_defaults_to_squad(self) -> None:
        wrapper = _load_module(
            "_squad_comparison_wrapper", SQUAD_COMPARISON_WRAPPER
        )
        assert wrapper.DEFAULT_DATASET == "squad"
        # The wrapper delegates to the generic script's main().
        assert wrapper.main.__module__ == "run_retriever_comparison"


class TestFigureOutputDir:
    def test_defaults_to_input_dir_name(self, plots_module) -> None:
        comparison_dir = Path("outputs/comparisons/squad_retrievers_semantic_local")
        out = plots_module.resolve_figure_output_dir(comparison_dir)
        assert out == Path("outputs/figures/squad_retrievers_semantic_local")

    def test_base_input_maps_to_base_figures(self, plots_module) -> None:
        comparison_dir = Path("outputs/comparisons/squad_retrievers_base")
        out = plots_module.resolve_figure_output_dir(comparison_dir)
        assert out == Path("outputs/figures/squad_retrievers_base")

    def test_figure_override_wins(self, plots_module, monkeypatch) -> None:
        monkeypatch.setenv(
            "RAGWATCH_FIGURE_OUTPUT_DIR", "outputs/figures/custom_semantic"
        )
        out = plots_module.resolve_figure_output_dir(
            Path("outputs/comparisons/squad_retrievers_semantic_local")
        )
        assert out == Path("outputs/figures/custom_semantic")

    def test_comparison_input_dir_default(self, plots_module) -> None:
        out = plots_module.resolve_comparison_input_dir()
        assert out == Path("outputs/comparisons/squad_retrievers_semantic_local")

    def test_comparison_input_dir_hotpotqa(self, plots_module, monkeypatch) -> None:
        monkeypatch.setenv("RAGWATCH_DATASET", "hotpotqa")
        out = plots_module.resolve_comparison_input_dir()
        assert out == Path(
            "outputs/comparisons/hotpotqa_retrievers_semantic_local"
        )

    def test_comparison_input_dir_override(self, plots_module, monkeypatch) -> None:
        monkeypatch.setenv(
            "RAGWATCH_COMPARISON_OUTPUT_DIR",
            "outputs/comparisons/squad_retrievers_base",
        )
        out = plots_module.resolve_comparison_input_dir()
        assert out == Path("outputs/comparisons/squad_retrievers_base")
