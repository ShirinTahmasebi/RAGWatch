"""Utility for loading Hugging Face datasets with a clear error on missing dependency."""


def load_hf_dataset(path: str, name: str | None = None, split: str = "validation"):
    """Load a Hugging Face dataset, raising a helpful error if `datasets` is not installed."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required for Hugging Face dataset adapters. "
            "Install it with: pip install ragwatch[datasets]"
        ) from e

    if name:
        return load_dataset(path, name, split=split)
    return load_dataset(path, split=split)
