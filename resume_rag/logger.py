"""Project logger. Rich console output for readable manual verification.

Usage:
    from resume_rag.logger import get_logger
    log = get_logger(__name__)
    log.info("retrieved %d chunks", len(chunks))
"""

from __future__ import annotations
import logging

from rich.logging import RichHandler
from resume_rag.config import get_settings

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = get_settings().log_level.upper()
    logging.basicConfig(
        level=level,
        format="%(name)s | %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )

    # quiet noisy thirdpart loggers
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True

def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)