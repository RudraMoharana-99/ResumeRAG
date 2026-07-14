"""Knowledge-base versioning for cache invalidation.

The version is a short hash derived from the resume files on disk + the
indexed child count. Re-ingest -> different inputs -> different hash ->
cache.get() treats all prior entries as stale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import get_settings
from src.logger import get_logger

log = get_logger(__name__)

VERSION_FILENAME = "kb_version.txt"


def _version_file_path() -> Path:
    """Where the current kb_version lives on disk."""
    settings = get_settings()
    return Path(settings.chroma_persist_dir) / VERSION_FILENAME


def compute_kb_version(child_count: int) -> str:
    """Deterministic 12-char hash of (resume filenames + sizes + child count).

    Same corpus -> same version. Add/remove/edit a resume -> version changes.
    """
    settings = get_settings()
    data_dir = Path(settings.raw_dir)

    inputs = sorted(
        (f.name, f.stat().st_size)
        for f in data_dir.iterdir()
        if f.suffix.lower() in {".pdf", ".docx"}
    )
    payload = json.dumps(inputs) + f"|children={child_count}"
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def write_kb_version(version: str) -> None:
    """Persist the current kb_version (called by the indexer after ingest)."""
    path = _version_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(version)
    log.info("kb_version written: %s -> %s", version, path)


def read_kb_version() -> str:
    """Read the current kb_version. Returns 'unversioned' if file missing."""
    path = _version_file_path()
    if not path.exists():
        return "unversioned"
    return path.read_text().strip()