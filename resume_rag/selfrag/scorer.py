"""Self-RAG scorer — scores a candidate with per-point cited evidence.

The LLM must attach an exact resume quote to every point. Those quotes are
later verified programmatically by the verifier (the LLM proposes, code checks).
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field
from resume_rag.components.llm import get_llm
from resume_rag.logger import get_logger

log = get_logger(__name__)


_SYSTEM = """You are a hiring assistant scoring ONE candidate against a recruiter query.

For each relevant point, output:
- type: "strength" or "gap"
- claim: a short factual statement
- evidence: an EXACT, VERBATIM quote copied from the resume that proves the claim.
  Copy it character-for-character. Do NOT paraphrase. For a "gap", evidence may
  be an empty string (you cannot quote the absence of something).

Rules:
- Every "strength" MUST have a verbatim evidence quote from the resume.
- Never invent qualifications. If the resume doesn't support a claim, don't make it.
- score: 0-100 overall fit for the query.
"""

class ScorePoint(BaseModel):
    type: Literal["strength", "gap"]
    claim: str = Field(description="Short factual stsatement")
    evidence: str = Field(description="Exact verbtim quote from resume (or empty for a gap)")


class CandidateScore(BaseModel):
    score: int = Field(ge=0, le=100, description="Overall fit 0-100")  # ge → Greater than or Equal to : le → Less than or Equal to
    summary: str = Field(description="One-sentence overall assessment")
    points: list[ScorePoint]


def score_candidate(query: str, resume_text: str) -> CandidateScore:
    """Score one candidate. resume_text should be the FULL resume."""
    scorer = get_llm().with_structured_output(CandidateScore)
    return scorer.invoke(
        [
            ("system", _SYSTEM),
            ("human", f"Recruiter query:\n{query}\n\nCandidate resume:\n{resume_text}"),
        ]
    )