"""Generate paper-quality comparison figures from SQuAD comparison outputs.

Requirements:
    pip install -e ".[plots]"

Usage:
    python examples/generate_squad_comparison_plots.py

By default, reads from the base comparison directory and writes figures to a
matching figures directory:
    outputs/comparisons/squad_retrievers_base/   ->   outputs/figures/squad_retrievers_base/

Set RAGWATCH_COMPARISON_OUTPUT_DIR to read from a different comparison directory
(for example a semantic run); the figure directory then mirrors its name unless
RAGWATCH_FIGURE_OUTPUT_DIR is also set:
    RAGWATCH_COMPARISON_OUTPUT_DIR=outputs/comparisons/squad_retrievers_semantic_local \
        python examples/generate_squad_comparison_plots.py

If the comparison outputs are missing, run this first:
    python examples/run_squad_retriever_comparison.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.config.env import get_env, load_env
from ragwatch.plots.comparison_plots import generate_all_comparison_plots

# Load .env so the optional override variables are available when present.
load_env()

# Optional overrides for the input comparison directory and the output figures
# directory. When unset, sensible mode-aware defaults are used.
ENV_COMPARISON_OUTPUT_DIR = "RAGWATCH_COMPARISON_OUTPUT_DIR"
ENV_FIGURE_OUTPUT_DIR = "RAGWATCH_FIGURE_OUTPUT_DIR"

DEFAULT_COMPARISON_DIR = Path("outputs/comparisons/squad_retrievers_base")
FIGURES_ROOT = Path("outputs/figures")


def resolve_comparison_input_dir() -> Path:
    """Resolve the comparison directory to read figures from."""
    override = get_env(ENV_COMPARISON_OUTPUT_DIR)
    if override:
        return Path(override)
    return DEFAULT_COMPARISON_DIR


def resolve_figure_output_dir(comparison_dir: Path) -> Path:
    """Resolve the figures output directory for a given comparison directory.

    A ``RAGWATCH_FIGURE_OUTPUT_DIR`` override, when set, is used verbatim.
    Otherwise figures are written under ``outputs/figures/<comparison name>`` so
    base and semantic figures do not overwrite each other.
    """
    override = get_env(ENV_FIGURE_OUTPUT_DIR)
    if override:
        return Path(override)
    return FIGURES_ROOT / comparison_dir.name


def main() -> None:
    comparison_dir = resolve_comparison_input_dir()
    output_dir = resolve_figure_output_dir(comparison_dir)

    if not (comparison_dir / "comparison_summary.csv").exists():
        print(
            "Comparison outputs not found in "
            f"{comparison_dir}.\n"
            "Generate them first by running:\n"
            "    python examples/run_squad_retriever_comparison.py"
        )
        return

    print(f"Reading comparison outputs from: {comparison_dir}")
    created = generate_all_comparison_plots(comparison_dir, output_dir)

    print(f"\nGenerated {len(created)} figures in {output_dir}:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
