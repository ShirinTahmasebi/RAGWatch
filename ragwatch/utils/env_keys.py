"""Canonical environment variable names used throughout RAGWatch."""
from __future__ import annotations


class EnvKeys:
    """String constants for environment variables required by RAGWatch."""

    LOG_DIR = "RAGWATCH_LOG_DIR"

    HOTPOTQA_LOG_DIR = "RAGWATCH_HOTPOTQA_LOG_DIR"
    HOTPOTQA_INDEX_DIR = "RAGWATCH_HOTPOTQA_INDEX_DIR"
    HOTPOTQA_SPLIT = "RAGWATCH_HOTPOTQA_SPLIT"
    HOTPOTQA_SAMPLE_SIZE = "RAGWATCH_HOTPOTQA_SAMPLE_SIZE"
    HOTPOTQA_USE_DUMMY_DATA = "RAGWATCH_HOTPOTQA_USE_DUMMY_DATA"

    @classmethod
    def all(cls) -> set[str]:
        return {
            cls.LOG_DIR,
            cls.HOTPOTQA_LOG_DIR,
            cls.HOTPOTQA_INDEX_DIR,
            cls.HOTPOTQA_SPLIT,
            cls.HOTPOTQA_SAMPLE_SIZE,
            cls.HOTPOTQA_USE_DUMMY_DATA,
        }
