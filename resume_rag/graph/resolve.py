"""Resolve candidate names to candidate_ids.

Used by both the compare path (2 names) and constrained-rank (3+ names).
Matches against candidate_texts.json: a name hits if it appears in the
resume's text (header/name line). Returns (resolved, unresolved).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from resume_rag.config import get_settings
from resume_rag.ingestion.candidate_store import FILENAME
from resume_rag.logger import get_logger

log = get_logger(__name__)


def _load_mapping() -> dict[str, str]:
    settings = get_settings()
    path = Path(settings.chroma_persist_dir).parent / FILENAME
    if not path.exists():
        raise RuntimeError("candidate_texts.json missing — re-run the indexer.")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def resolve_names(names: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map each name to a candidate_id.

    Returns:
        (resolved: {name: candidate_id}, unresolved: [names not found])
    """
    mapping = _load_mapping()
    norm_texts = {cid: _normalize(text) for cid, text in mapping.items()}

    resolved: dict[str, str] = {}
    unresolved: list[str] = []

    for name in names:
        n = _normalize(name)
        # match full name, or all name tokens present near the top of the resume
        hit = None
        for cid, text in norm_texts.items():
            if n in text[:400]:           # name usually in the header region
                hit = cid
                break
        if hit:
            resolved[name] = hit
        else:
            unresolved.append(name)

    log.info("resolve_names: resolved=%s unresolved=%s", resolved, unresolved)
    return resolved, unresolved