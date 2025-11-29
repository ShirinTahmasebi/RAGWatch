"""Centralized environment helpers for RAGWatch."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable '{name}' is not set")
    return value


def env_str(name: str) -> str:
    """Return a required environment variable as a string."""

    return _require_env(name)


def env_path(name: str) -> Path:
    """Return a required environment variable interpreted as a path."""

    return Path(_require_env(name)).expanduser()
