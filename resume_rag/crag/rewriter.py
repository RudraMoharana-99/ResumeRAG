"""CRAG query rewriter - used in the corrective step when retrieval is weak."""

from __future__ import annotations

from resume_rag.components.llm import get_llm
from resume_rag.logger import get_logger

log = get_logger(__name__)

_SYSTEM = """Rewrite the recruiter's query to improve retrieval over a resume \
    
database. Expand abbreviations, add synonyms for skills and job titles, and make \
implicit requirements explicit. Return ONLY the rewritten query - no preamble. \
Return a single focused query of at most 20 words. Do not produce OR-lists or keyword dumps. \
"""


def rewrite_query(query: str) -> str:
    """Return an expanded version of the query for re-retrieval"""
    result = get_llm().invoke(
        [
            ("system", _SYSTEM),
            ("human", query),
        ]
    )
    return result.content.strip()
