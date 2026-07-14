"""Self-RAG grounding verifier - programmatic ISSUP check.

Does each cited quote actually appear in the candidate's resume? We normalize
whitespace + case so trivial formatting differences don't cause false flags,
but we do NOT fuzzy-match — the quote must really be there.
"""

from __future__ import annotations
import re

from src.logger import get_logger
from src.selfrag.scorer import ScorePoint

log = get_logger(__name__)


def _normalize(text: str) -> str:
    """Lowercase + collapse all whitespace to single spaces."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_grounded(point: ScorePoint, resume_text: str) -> bool:
    """True if the point's evidence quote exists in the resume.
    
    Gaps with empty evidence are considered grounded (nothing to verify).
    Strengths with empty evidence are NEVER grounded.
    """
    if point.type == "gap" and not point.evidence.strip():
        return True
    if not point.evidence.strip():
        return False

    return _normalize(point.evidence) in _normalize(resume_text)


def verify_points(
        points: list[ScorePoint], resume_text: str
) -> tuple[list[ScorePoint], list[ScorePoint]]:
    """Split points into (grounded, ungrounded)."""
    grounded, ungrounded = [], []

    for p in points:
        (grounded if is_grounded(p, resume_text) else ungrounded).append(p)

    return grounded, ungrounded