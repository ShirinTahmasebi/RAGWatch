"""Backward-compatible SQuAD drift plots (thin wrapper).

The dataset-agnostic entry point examples/generate_drift_plots.py is the
preferred plot script. This wrapper simply defaults the dataset to SQuAD so the
historical command keeps working:

    python examples/generate_squad_drift_plots.py

It is equivalent to:

    RAGWATCH_DATASET=squad python examples/generate_drift_plots.py

All behavior (dataset-/mode-aware default input directory,
RAGWATCH_DRIFT_OUTPUT_DIR / RAGWATCH_DRIFT_FIGURE_OUTPUT_DIR overrides) is
inherited from the generic script.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "examples"))

from generate_drift_plots import main  # noqa: E402

DEFAULT_DATASET = "squad"


if __name__ == "__main__":
    main(default_dataset=DEFAULT_DATASET)
