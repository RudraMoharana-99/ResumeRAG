"""Intent router  classifies a recruiter query into a workflow

LLM structured output, not keyword matching, so it's robust to phrasing.
"""

from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

from resume_rag.components.llm import get_llm

from resume_rag.logger import get_logger

log = get_logger(__name__)


_SYSTEM = """Classify the recruiter's request into exactly one intent:

- "rank": find/rank/shortlist candidates for a role, skill, or job description.
  Examples: "top candidates for a data analyst role", "who knows SQL", "best fit for this JD".

- "compare": a direct head-to-head comparison of EXACTLY TWO named candidates.
  Examples: "compare Manish vs Lopamudra", "who is stronger, Alice or Bob".

Rules:
- List EVERY candidate name mentioned in "names".
- Choose "compare" ONLY for a two-person head-to-head (vs / versus / compare X and Y).
- If three or more people are named, choose "rank" (a side-by-side leaderboard).
- If the phrasing is "score/shortlist these people" rather than head-to-head, choose "rank".
"""


class Intent(BaseModel):
    intent: Literal["rank", "compare"] = Field(description="The workflow to run")
    names: list[str] = Field(
        default_factory=list,
        description="EVERY candidate name mentioned. Empty for an open rank query.",
    )


def classify(query: str) -> Intent:
    router = get_llm().with_structured_output(Intent)
    result = router.invoke([("system", _SYSTEM), ("human", query)])
    log.info("router: intent=%s names=%s", result.intent, result.names)
    return result