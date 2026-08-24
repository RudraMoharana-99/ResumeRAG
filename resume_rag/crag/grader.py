"""CRAG relevance grader - strict per-document yes/no via structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field

from resume_rag.components.llm import get_llm
from resume_rag.logger import get_logger


log = get_logger(__name__)


_SYSTEM = """You grade whether a resume excerpt is RELEVANT to a recruiter's query.

Relevant = the excerpt contains skills, experience, or qualifications that \
genuinely match what the query asks for.

Be STRICT. A loose topical association is NOT relevant. Only mark relevant=true \
if this excerpt would actually help a recruiter answer the query. When unsure, \
mark relevant=false."""


class RelevanceGrade(BaseModel):
    relevant: bool = Field(description="True only if genuinely relevant to the query")
    reason: str = Field(description="One short sentence justifying the grade")


def grade_document(query: str, doc_text: str) -> RelevanceGrade:
    """Grade a single document's relevance to the query."""
    grader = get_llm().with_structured_output(RelevanceGrade)

    return grader.invoke([
        ("system", _SYSTEM),
        ("human", f"Query:\n{query}\n\nResume excerpt:\n{doc_text}\n\nIs it relevant?")
    ])

