"""Verify indexer: ingest, check counts, then test parent-child retrieval."""

from resume_rag.components.vectorstore import (
    get_child_vectorstore,
    get_parent_child_retriever,
    get_parent_docstore,
)
from resume_rag.config import get_settings
from resume_rag.ingestion.chunker import chunk_all_resumes
from resume_rag.ingestion.indexer import index_resumes
from resume_rag.ingestion.loader import load_all_resumes


def main():
    settings = get_settings()

    # ── 1. Load + chunk ───────────────────────────────────────────────────────
    print("Loading resumes...")
    resumes = load_all_resumes(settings.raw_dir)
    print(f"  {len(resumes)} resumes loaded.")

    parents, children = chunk_all_resumes(resumes)
    print(f"  {len(parents)} parents, {len(children)} children.\n")

    # ── 2. Index ──────────────────────────────────────────────────────────────
    index_resumes(parents, children, resumes=resumes)

    # ── 3. Verify counts ──────────────────────────────────────────────────────
    child_store = get_child_vectorstore()
    parent_docstore = get_parent_docstore()

    child_count = child_store._collection.count()
    parent_keys = list(parent_docstore.yield_keys())

    print(f"\nStorage counts:")
    print(f"  resume_children (Chroma)  : {child_count}")
    print(f"  parent docstore (keys)    : {len(parent_keys)}")

    # ── 4. Test parent-child retrieval ────────────────────────────────────────
    query = "Microbiologist with QC experience"
    print(f"\nTest query: {query!r}")

    retriever = get_parent_child_retriever(k=3)
    parent_results = retriever.invoke(query)

    print(f"\nRetriever returned {len(parent_results)} parent docs:")
    for i, doc in enumerate(parent_results):
        print(f"\n  result {i+1}")
        print(f"  candidate_id : {doc.metadata['candidate_id']}")
        print(f"  source       : {doc.metadata['source']}")
        print(f"  parent_id    : {doc.metadata['parent_id']}")
        print(f"  parent length: {len(doc.page_content)} chars")
        print(f"  preview      : {doc.page_content[:200]!r}")


if __name__ == "__main__":
    main()