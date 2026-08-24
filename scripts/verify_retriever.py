"""Verify Layer 0: hybrid retrieval -> rerank -> parents, on 3 query types."""

from resume_rag.components.retriever import HybridRetriever

def show(title: str, parents):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    for i, doc in enumerate(parents):
        print(f"\n  parent {i+1}")
        print(f"  candidate_id: {doc.metadata['candidate_id']}")
        print(f"  source      : {doc.metadata['source']}")
        print(f"  preview     : {doc.page_content[:180]!r}")


def main():
    retriever = HybridRetriever()

    # ── 1. Semantic / paraphrase query (vector should shine) ─────────────────
    parents = retriever.retrieve("candidates who managed people or led teams")
    show("Query 1 (paraphrase): 'managed people or led teams'", parents)

    # ── 2. Exact-token query (BM25 should shine) ─────────────────────────────
    parents = retriever.retrieve("SQL")
    show("Query 2 (exact token): 'SQL'", parents)

    # ── 3. Metadata-filtered query ────────────────────────────────────────────
    if parents:
        cid = parents[0].metadata["candidate_id"]
        filtered = retriever.retrieve("technical skills", filter={"candidate_id": cid})
        show(f"Query 3 (filtered to candidate_id={cid}): 'technical skills'", filtered)
        assert all(p.metadata["candidate_id"] == cid for p in filtered), \
            "filter leaked other candidates!"
        print("\n  filter assertion passed ✓")

    # ── 4. RRF fallback path ──────────────────────────────────────────────────
    rrf_retriever = HybridRetriever(use_rrf=True)
    parents = rrf_retriever.retrieve("python machine learning experience")
    show("Query 4 (RRF path, no Cohere call): 'python machine learning'", parents)


if __name__ == "__main__":
    main()