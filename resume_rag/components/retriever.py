"""Hybrid retriever (Layer 0 - Advanced RAG).

Pipeline:
    vector top-k (Chroma children) + BM25 top-k (children)
        -> union (dedupe by content)
        -> Cohere rerank (default) OR RRF fusion (fallback, no API cost)
        -> resolve parents via docstore
        -> deduped parent Documents

RRF fallback exits for when Cohere quota/latency is an issue:
    HybridRetriever(use_rrf=True)
"""
from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from resume_rag.components.reranker import get_reranker
from resume_rag.components.vectorstore import (
    PARENT_ID_KEY,
    get_child_vectorstore,
    get_parent_docstore,
)
from resume_rag.config import get_settings
from resume_rag.logger import get_logger
import re


log = get_logger(__name__)

RRF_K = 60


def _load_all_children() -> list[Document]:
    """Pull every child chunk out of Chroma to build the BM25 index."""
    store = get_child_vectorstore()
    raw = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


class HybridRetriever:
    def __init__(self, use_rrf: bool = False) -> None:
        settings = get_settings()
        self._k = settings.retrieval_k
        self._top_n = settings.rerank_top_n
        self._use_rrf = use_rrf
        self.rrf_k = settings.rrf_k

        self._child_store = get_child_vectorstore()
        self._parent_docstore = get_parent_docstore()

        children = _load_all_children()

        if not children:
            raise RuntimeError("No Children in Chroma - run the indexer first.")

        self._bm25 = BM25Retriever.from_documents(children, k=self._k, preprocess_func=self._bm25_preprocess,)
        log.info(
            "HybridRetrieever ready: %d children indexed from BM25 (use_rrf=%s)",
            len(children), use_rrf
        )

    # ── public ────────────────────────────────────────────────────────────────

    def retrieve(
            self,
            query: str,
            filter: dict | None = None,
    ) -> list[Document]:
        """Full pipeline. Returns deduped parent Documents, best-first.
        
        Args:
            query: recruiter query / JD text
            filter: optional Chroma-style metadata filter, e.g.
                    {"candidate_id": "ab12cd34}
        """
        #1. Vector search (children)
        vector_hits = self._child_store.similarity_search(
            query=query, k=self._k, filter=filter
        )

        #2. BM25 search (children) - filter applied post-hoc
        bm25_hits = self._bm25.invoke(query)
        if filter:
            bm25_hits = [
                d for d in bm25_hits
                if all(d.metadata.get(k) == v for k, v in filter.items())
            ]

        log.info("vector hits=%d    bm25 hits=%d", len(vector_hits), len(bm25_hits))

        # 3. Fuse/select children
        if self._use_rrf:
            top_children = self._rrf_fuse(vector_hits, bm25_hits)[:self._top_n]
        else:
            pool = self._dedupe(vector_hits + bm25_hits)
            log.info("merged pool=%d children", len(pool))
            top_children = get_reranker().compress_documents(documents=pool, query=query)

        log.info("selected %d children", len(top_children))

        # 4. Resolve to parents (dedupe, preserve order)
        return self._resolve_parents(top_children)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe(docs: list[Document]) -> list[Document]:
        seen: set[str] = set()
        out = []

        for d in docs:
            key = d.page_content
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    @staticmethod
    def _rrf_fuse(*ranked_lists: list[Document]) -> list[Document]:
        """Reciprocal Rank Fusion: score = sum over lists of 1/(rank + RRF_K)."""
        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content
                by_key[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank+ 1 + RRF_K)

        ordered = sorted(scores, key=scores.get, reverse=True)
        return [by_key[k] for k in ordered]


    def _resolve_parents(self, children: list[Document]) -> list[Document]:
        parent_ids: list[str] = []
        for c in children:
            pid = c.metadata.get(PARENT_ID_KEY)
            if pid and pid not in parent_ids:
                parent_ids.append(pid)

        parents = self._parent_docstore.mget(parent_ids)
        out = [p for p in parents if p is not None]

        log.info("resolved %d unique parents", len(out))

        return out

    @staticmethod
    def _bm25_preprocess(text: str) -> list[str]:
        """Lowercase + alphanumeric tokens so 'SQL' matches 'SQL,' / 'sql'."""
        return re.findall(r"[a-z0-9+#.]+", text.lower())