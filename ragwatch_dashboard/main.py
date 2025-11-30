"""Streamlit entry point for the Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path
import sys

# Ensure the repository root (which contains the `ragwatch` package) is importable even
# when Streamlit launches this script from the `ragwatch_dashboard` directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from controller.dashboard_controller import get_controller  # noqa: E402
from ui.dashboard_view import render_dashboard  # noqa: E402

controller = get_controller()
render_dashboard(controller)
