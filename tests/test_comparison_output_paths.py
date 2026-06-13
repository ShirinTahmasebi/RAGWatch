"""Tests for mode-aware comparison/figure output directory resolution.

These tests exercise the small helper functions in the comparison and plot
example scripts. They never require Hugging Face, Chroma, Qdrant, pgvector,
Postgres, OpenAI, Azure, Streamlit, or Plotly.
"""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
COMPARISON_EXAMPLE = EXAMPLES_DIR / "run_squad_retriever_comparison.py"
PLOTS_EXAMPLE = EXAMPLES_DIR / "generate_squad_comparison_plots.py"


def _load_module(name: str, path: Path):
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
    """Ensure output-dir overrides are unset unless a test sets them."""
    monkeypatch.delenv("RAGWATCH_COMPARISON_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("RAGWATCH_FIGURE_OUTPUT_DIR", raising=False)


class TestComparisonOutputDir:
    def test_base_mode_resolves_to_base_dir(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            compute_semantic_kpis=False,
            semantic_provider_name=None,
            semantic_enabled_successfully=False,
        )
        assert out == Path("outputs/comparisons/squad_retrievers_base")

    def test_semantic_local_resolves(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            compute_semantic_kpis=True,
            semantic_provider_name="local",
            semantic_enabled_successfully=True,
        )
        assert out == Path("outputs/comparisons/squad_retrievers_semantic_local")

    def test_semantic_openai_resolves(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            compute_semantic_kpis=True,
            semantic_provider_name="openai",
            semantic_enabled_successfully=True,
        )
        assert out == Path("outputs/comparisons/squad_retrievers_semantic_openai")

    def test_semantic_azure_resolves(self, comparison_module) -> None:
        out = comparison_module.resolve_comparison_output_dir(
            compute_semantic_kpis=True,
            semantic_provider_name="azure_openai",
            semantic_enabled_successfully=True,
        )
        assert out == Path(
            "outputs/comparisons/squad_retrievers_semantic_azure_openai"
        )

    def test_semantic_requested_but_failed_uses_base(
        self, comparison_module
    ) -> None:
        # Semantic requested, but provider setup failed: no semantic columns,
        # so the base directory is used.
        out = comparison_module.resolve_comparison_output_dir(
            compute_semantic_kpis=True,
            semantic_provider_name="openai",
            semantic_enabled_successfully=False,
        )
        assert out == Path("outputs/comparisons/squad_retrievers_base")

    def test_override_wins(self, comparison_module, monkeypatch, capsys) -> None:
        monkeypatch.setenv(
            "RAGWATCH_COMPARISON_OUTPUT_DIR", "outputs/comparisons/my_custom_run"
        )
        out = comparison_module.resolve_comparison_output_dir(
            compute_semantic_kpis=True,
            semantic_provider_name="local",
            semantic_enabled_successfully=True,
        )
        assert out == Path("outputs/comparisons/my_custom_run")
        assert "custom comparison output directory" in capsys.readouterr().out.lower()


class TestComparisonModeName:
    def test_base_name(self, comparison_module) -> None:
        assert (
            comparison_module.resolve_comparison_mode_name(False, None)
            == "squad_retrievers_base"
        )

    def test_semantic_name(self, comparison_module) -> None:
        assert (
            comparison_module.resolve_comparison_mode_name(True, "local")
            == "squad_retrievers_semantic_local"
        )


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
        assert out == Path("outputs/comparisons/squad_retrievers_base")

    def test_comparison_input_dir_override(self, plots_module, monkeypatch) -> None:
        monkeypatch.setenv(
            "RAGWATCH_COMPARISON_OUTPUT_DIR",
            "outputs/comparisons/squad_retrievers_semantic_local",
        )
        out = plots_module.resolve_comparison_input_dir()
        assert out == Path("outputs/comparisons/squad_retrievers_semantic_local")
