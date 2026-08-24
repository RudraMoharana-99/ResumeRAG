"""Indexer — children into Chroma, parents into disk-backed docstore."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from resume_rag.cache.kb_version import compute_kb_version, write_kb_version
from resume_rag.components.vectorstore import (
    PARENT_ID_KEY,
    get_child_vectorstore,
    get_parent_docstore,
)
from resume_rag.ingestion.candidate_store import write_candidate_texts
from resume_rag.config import get_settings
from resume_rag.logger import get_logger

log = get_logger(__name__)


def _child_id(doc: Document, idx: int) -> str:
    """Stable ID for a child chunk."""
    return f"{doc.metadata['parent_id']}_c{idx}"


def index_resumes(
        parents: list[Document],
        children: list[Document],
        resumes: list[dict] | None = None,
) -> None:
    """Upsert children into Chroma and parents into the docstore.

    Children: embedded + searchable.
    Parents:  stored by parent_id (matches the id_key the retriever uses).
    """
    settings = get_settings()

    log.info("Indexing %d children into 'resume_children'...", len(children))
    child_store = get_child_vectorstore()
    child_ids = [_child_id(doc=doc, idx=idx) for idx, doc in enumerate(children)]
    child_store.add_documents(documents=children, ids=child_ids)

    # ── 2. Parents -> docstore (keyed by parent_id) ──────────────────────────
    log.info("Indexing %d parents into 'resume_parents'...", len(parents))
    parents_docstore = get_parent_docstore()
    parent_pairs = [(p.metadata[PARENT_ID_KEY], p) for p in parents]
    parents_docstore.mset(parent_pairs)

    # ── 3. Write kb_version (cache invalidation signal) ──────────────────────
    version = compute_kb_version(child_count=len(children))
    write_kb_version(version)

    # ── 4. Candidate full-text artifact (for Self-RAG grounding) ─────────────
    if resumes is not None:
        write_candidate_texts(resumes=resumes)

    log.info("Indexing complete. Chroma presisted at: %s", settings.chroma_persist_dir)
