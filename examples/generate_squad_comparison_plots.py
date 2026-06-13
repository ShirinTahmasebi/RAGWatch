"""Generate paper-quality comparison figures from SQuAD comparison outputs.

Requirements:
    pip install -e ".[plots]"

Usage:
    python examples/generate_squad_comparison_plots.py

Reads comparison outputs from:
    outputs/comparisons/squad_retrievers/

Saves figures to:
    outputs/figures/squad_retrievers/

If the comparison outputs are missing, run this first:
    python examples/run_squad_retriever_comparison.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.plots.comparison_plots import generate_all_comparison_plots


def main() -> None:
    comparison_dir = Path("outputs/comparisons/squad_retrievers")
    output_dir = Path("outputs/figures/squad_retrievers")

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
