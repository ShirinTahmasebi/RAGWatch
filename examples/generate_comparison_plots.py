"""Generate paper-quality comparison figures from retriever-comparison outputs.

This is the dataset-agnostic plot entry point. The default input directory is
derived from ``RAGWATCH_DATASET`` and the semantic mode, mirroring
examples/run_retriever_comparison.py:

    RAGWATCH_DATASET=squad    python examples/generate_comparison_plots.py
    RAGWATCH_DATASET=hotpotqa python examples/generate_comparison_plots.py

Requirements:
    pip install -e ".[plots]"

By default, reads from the dataset-/mode-aware comparison directory and writes
figures to a matching figures directory:
    outputs/comparisons/squad_retrievers_semantic_local/
        ->  outputs/figures/squad_retrievers_semantic_local/

Set RAGWATCH_SEMANTIC_PROVIDER / RAGWATCH_DISABLE_SEMANTIC_KPIS to match the run
mode. Set RAGWATCH_COMPARISON_OUTPUT_DIR to read from a different comparison
directory verbatim; the figure directory then mirrors its name unless
RAGWATCH_FIGURE_OUTPUT_DIR is also set.

If the comparison outputs are missing, run this first:
    python examples/run_retriever_comparison.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.config.env import get_env, load_env
from ragwatch.datasets.registry import dataset_name_from_env
from ragwatch.experiments.output_naming import comparison_output_dir
from ragwatch.plots.comparison_plots import generate_all_comparison_plots
from ragwatch.semantic.providers import (
    resolve_semantic_provider_name,
    semantic_kpis_enabled,
)

# Load .env so the optional override variables are available when present.
load_env()

DEFAULT_DATASET = "squad"

# Optional overrides for the input comparison directory and the output figures
# directory. When unset, dataset-/mode-aware defaults are used.
ENV_COMPARISON_OUTPUT_DIR = "RAGWATCH_COMPARISON_OUTPUT_DIR"
ENV_FIGURE_OUTPUT_DIR = "RAGWATCH_FIGURE_OUTPUT_DIR"

FIGURES_ROOT = Path("outputs/figures")


def resolve_comparison_input_dir(default_dataset: str = DEFAULT_DATASET) -> Path:
    """Resolve the comparison directory to read figures from.

    A ``RAGWATCH_COMPARISON_OUTPUT_DIR`` override, when set, is used verbatim.
    Otherwise the directory is derived from the dataset name and semantic mode.
    """
    override = get_env(ENV_COMPARISON_OUTPUT_DIR)
    if override:
        return Path(override)
    dataset_name = dataset_name_from_env(default_dataset)
    semantic_enabled = semantic_kpis_enabled()
    provider_name = (
        resolve_semantic_provider_name() if semantic_enabled else None
    )
    return comparison_output_dir(dataset_name, semantic_enabled, provider_name)


def resolve_figure_output_dir(comparison_dir: Path) -> Path:
    """Resolve the figures output directory for a given comparison directory.

    A ``RAGWATCH_FIGURE_OUTPUT_DIR`` override, when set, is used verbatim.
    Otherwise figures are written under ``outputs/figures/<comparison name>`` so
    datasets and modes do not overwrite each other.
    """
    override = get_env(ENV_FIGURE_OUTPUT_DIR)
    if override:
        return Path(override)
    return FIGURES_ROOT / comparison_dir.name


def main(default_dataset: str = DEFAULT_DATASET) -> None:
    comparison_dir = resolve_comparison_input_dir(default_dataset)
    output_dir = resolve_figure_output_dir(comparison_dir)

    if not (comparison_dir / "comparison_summary.csv").exists():
        print(
            "Comparison outputs not found in "
            f"{comparison_dir}.\n"
            "Generate them first by running:\n"
            "    python examples/run_retriever_comparison.py"
        )
        return

    print(f"Reading comparison outputs from: {comparison_dir}")
    created = generate_all_comparison_plots(comparison_dir, output_dir)

    print(f"\nGenerated {len(created)} figures in {output_dir}:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
