"""Launch the RAGWatch research dashboard.

Requirements:
    pip install -e ".[dashboard]"

First generate comparison outputs:
    python examples/run_squad_retriever_comparison.py

Then launch the dashboard (either command works):
    streamlit run src/ragwatch/dashboard/app.py
    python examples/run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    app_path = project_root / "src" / "ragwatch" / "dashboard" / "app.py"

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. Install the dashboard extra:")
        print('  pip install -e ".[dashboard]"')
        sys.exit(1)

    subprocess.run(["streamlit", "run", str(app_path)], check=False)


if __name__ == "__main__":
    main()
