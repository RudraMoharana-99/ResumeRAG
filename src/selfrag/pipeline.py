"""Self-RAG orchestration (Layer 2): score -> verify -> regenerate (bounded).

If any cited claim is ungrounded, regenerate. On retry exhaustion, keep only
the grounded claims and flag that some were dropped — never ship a fabrication.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import get_settings
from src.ingestion.candidate_store import load_candidate_text
from src.logger import get_logger
from src.selfrag.scorer import CandidateScore, ScorePoint, score_candidate
from src.selfrag.verifier import verify_points

log = get_logger(__name__)


@dataclass
class VerifiedScore:
    candidate_id: str
    score: int
    summary: str
    grounded_points: list[ScorePoint]
    dropped_points: list[ScorePoint]      # ungrounded, remove after retries
    attempts: int
    fully_grounded: bool


class SelfRAGPipeline:
    def __init__(self, max_retries: int | None = None) -> None:
        settings = get_settings()
        self._max_retries = (
            max_retries if max_retries is not None else settings.selfrag_max_retries
        )

    def run(self, query: str, candidate_id: str) -> VerifiedScore:
        resume_text = load_candidate_text(candidate_id=candidate_id)
        if resume_text is None:
            raise RuntimeError(
                f"No full text for candidate_id={candidate_id!r}. Re-run the indexer."
            )

        last: CandidateScore | None = None
        grounded: list[ScorePoint] = []
        ungrounded: list[ScorePoint] = []

        for attempt in range(1, self._max_retries+2): # 1 intial + N retries
            last = score_candidate(query=query, resume_text=resume_text)
            grounded, ungrounded = verify_points(last.points, resume_text=resume_text)
            log.info(
                "candidate=%s attempt=%d: %d grounded, %d ungrounded",
                candidate_id, attempt, len(grounded), len(ungrounded),
            )
            if not ungrounded:
                return VerifiedScore(
                    candidate_id=candidate_id, score=last.score,
                    summary=last.summary, grounded_points=grounded,
                    dropped_points=[], attempts=attempt, fully_grounded=True,
                )
        # Retries exhausted — keep only grounded claims, flag the rest
        log.warning(
            "candidate=%s: %d ungrounded claims dropped after %d attempts",
            candidate_id, len(ungrounded), attempt,
        )
        return VerifiedScore(
            candidate_id=candidate_id, score=last.score,
            summary=last.summary, grounded_points=grounded,
            dropped_points=ungrounded, attempts=attempt,
            fully_grounded=False,
        )

