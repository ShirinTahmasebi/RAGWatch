"""Thin wrapper that behaves like a mutable mapping over Streamlit session state."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


class SessionStateAdapter(MutableMapping):
    def __init__(self, backing) -> None:
        self._backing = backing

    def __getitem__(self, key: str) -> Any:
        return self._backing[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._backing[key] = value

    def __delitem__(self, key: str) -> None:
        del self._backing[key]

    def __iter__(self):
        return iter(self._backing)

    def __len__(self) -> int:
        return len(self._backing)

    def get(self, key: str, default: Any = None) -> Any:
        return self._backing.get(key, default)
