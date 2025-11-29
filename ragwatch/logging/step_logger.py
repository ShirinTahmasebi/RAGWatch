"""Console-friendly step logger reused across RAGWatch components."""
from __future__ import annotations

import logging
from typing import Dict, Optional

try:  # pragma: no cover - colorama is an optional nicety.
    from colorama import Fore, Style, init as colorama_init
except ImportError:  # pragma: no cover
    class _FallbackColor:  # type: ignore[too-many-ancestors]
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
        LIGHTBLUE_EX = LIGHTCYAN_EX = LIGHTGREEN_EX = ""

    Fore = _FallbackColor()  # type: ignore[assignment]
    Style = _FallbackColor()  # type: ignore[assignment]

    def colorama_init(*_args, **_kwargs):  # type: ignore[override]
        return None

colorama_init(autoreset=False)

_DEFAULT_STAGE_COLORS: Dict[str, str] = {
    "prep": Fore.CYAN,
    "vector": Fore.MAGENTA,
    "retriever": Fore.BLUE,
    "rag": Fore.CYAN,
    "stream": Fore.YELLOW,
    "session": Fore.GREEN,
    "retrieval": Fore.LIGHTBLUE_EX,
    "answer": Fore.LIGHTCYAN_EX,
    "write": Fore.WHITE,
    "error": Fore.RED,
}


class ConsoleStepLogger:
    """Write human-readable progress updates to stdout with color hints."""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        stage_colors: Optional[Dict[str, str]] = None,
    ) -> None:
        self._logger = logger or self._init_logger()
        self._stage_colors = stage_colors or dict(_DEFAULT_STAGE_COLORS)

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger("ragwatch.steps")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[RAGWatch] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def log(
        self,
        stage: str,
        message: str,
        *args,
        level: int = logging.INFO,
        exc_info: bool | BaseException | tuple[BaseException, BaseException, BaseException] | None = False,
    ) -> None:
        color = self._stage_colors.get(stage, "")
        reset = Style.RESET_ALL if color else ""
        prefix = f"[{stage.upper()}] "
        fmt = f"{color}{prefix}{message}{reset}"
        self._logger.log(level, fmt, *args, exc_info=exc_info)