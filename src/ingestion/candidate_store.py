"""Candidate full-text store - an ingestion side-artifact.

Self-RAG verifies that every cited quote actually exist in the candidate's
resume. Grounding is checked against the FULL resume (source of truth), not
just the chunk we happened to retrieve. This modules write and read that lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from src.config import get_settings
from src.logger import get_logger


log = get_logger(__name__)

FILENAME = "candidate_texts.json"

def _path() -> Path:
    settings = get_settings()
    return Path(settings.chroma_persist_dir).parent / FILENAME


def write_candidate_texts(resumes: list[dict]) -> None:
    """Persist {candidate_id: full_text} from loader output."""

    mapping = {r['candidate_id']: r['text'] for r in resumes}
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    log.info("candidate_texts written: %d candidates -> %s", len(mapping), path)


def load_candidate_text(candidate_id: str) -> str | None:
    """Return full resume text for a candidate, or None if absent."""
    path = _path()
    if not path.exists():
        return None
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return mapping.get(candidate_id)
