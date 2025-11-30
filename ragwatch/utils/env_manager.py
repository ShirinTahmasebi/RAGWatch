"""Centralized environment helpers for RAGWatch."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

from .env_keys import EnvKeys

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


def ensure_env_vars(*names: str | Iterable[str]) -> None:
    """Validate that all provided environment variables are set."""

    flattened: list[str] = []
    for item in names:
        if isinstance(item, str):
            flattened.append(item)
        else:
            flattened.extend(list(item))

    missing = [name for name in flattened if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(sorted(set(missing)))
        )


def resolve_log_dir(
    *,
    override: Optional[str],
    dataset_env_var: Optional[str] = None,
) -> str:
    """Resolve the directory where RAG runs should be logged."""

    if override:
        return override

    if dataset_env_var:
        try:
            return env_str(dataset_env_var)
        except RuntimeError:
            pass

    ensure_env_vars(EnvKeys.LOG_DIR)
    return env_str(EnvKeys.LOG_DIR)
