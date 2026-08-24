"""CRAG orechetration (Layer 1).

retrieve -> grade -> enough relevent? -> done
                                      -> no -> rewrite -> re-retrieve -> re-grade
                                                        -> done | no_strong_match
"""

from __future__ import annotations
from dataclasses import dataclass

from langchain_core.documents import Document

from resume_rag.components.retriever import HybridRetriever
from resume_rag.config import get_settings
from resume_rag.crag.grader import RelevanceGrade, grade_document
from resume_rag.crag.rewriter import rewrite_query
from resume_rag.logger import get_logger

log = get_logger(__name__)


@dataclass
class CRAGResult:
    status: str                    # "strong_match | "no strong_match"
    query_used: str                # final query (may be rewritten)
    rewritten: bool
    relevant_docs: list[Document]
    graded: list[tuple[Document, RelevanceGrade]]  # all (doc, grade) for inspection


class CRAGPipeline:
    def __init__(self, min_relevant: int | None = None) -> None:
        settings = get_settings()
        self._retriever = HybridRetriever()
        self._min_relevant = min_relevant or settings.crag_min_relevant

    def run(self, query: str) -> CRAGResult:
        # ── Round 1 ───────────────────────────────────────────────────────────
        docs = self._retriever.retrieve(query)
        relevant, graded = self._grade_all(query, docs)
        log.info("round 1: %d/%d relevant", len(relevant), len(docs))

        if len(relevant) >= self._min_relevant:
            return CRAGResult(
                status="strong_match",
                query_used=query,
                rewritten= False,
                relevant_docs=relevant,
                graded=graded,
            )

        # ── Corrective step: rewrite + re-retrieve ───────────────────────────
        new_query = rewrite_query(query=query)
        log.info("rewrite: %r -> %r", query, new_query)

        docs2 = self._retriever.retrieve(new_query)
        relevant2, graded2 = self._grade_all(new_query, docs2)
        log.info("round 2: %d/%d relevant", len(relevant2), len(docs2))

        if len(relevant2) >= self._min_relevant:
            return CRAGResult(
                status="strong_match",
                query_used=new_query,
                rewritten=True,
                relevant_docs=relevant2,
                graded=graded2,
            )
        return CRAGResult(
            status="no_strong_match",
            query_used=new_query,
            rewritten=True,
            relevant_docs=[],
            graded=graded2,
        )


    # ── internals ─────────────────────────────────────────────────────────────
    
    def _grade_all(
            self, query: str, docs: list[Document]
    ) -> tuple[list[Document], list[tuple[Document, RelevanceGrade]]]:
        graded: list[tuple[Document, RelevanceGrade]] = []
        relevant: list[Document] = []

        for d in docs:
            grade = grade_document(query=query, doc_text=d.page_content)
            graded.append((d, grade))
            if grade.relevant:
                relevant.append(d)
        return relevant, graded
