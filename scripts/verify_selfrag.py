"""Verify Self-RAG: score a candidate, show grounding verdict per point.

Includes a planted-fabrication test: we monkeypatch the scorer to inject a
fake quote and confirm the verifier catches it.
"""

from resume_rag.crag.pipeline import CRAGPipeline
from resume_rag.selfrag.pipeline import SelfRAGPipeline
from resume_rag.selfrag import verifier
from resume_rag.selfrag.scorer import ScorePoint


def show(result):
    print(f"\n{'═'*60}")
    print(f"candidate_id : {result.candidate_id}")
    print(f"score        : {result.score}/100")
    print(f"summary      : {result.summary}")
    print(f"attempts     : {result.attempts}   fully_grounded: {result.fully_grounded}")
    print(f"\ngrounded points ({len(result.grounded_points)}):")
    for p in result.grounded_points:
        print(f"  [{p.type}] {p.claim}")
        if p.evidence:
            print(f"      ↳ evidence: {p.evidence[:90]!r}")
    if result.dropped_points:
        print(f"\ndropped (ungrounded) points ({len(result.dropped_points)}):")
        for p in result.dropped_points:
            print(f"  [{p.type}] {p.claim}  ✗ quote not in resume")


def main():
    query = "data analyst with SQL experience"

    # Pull a real candidate via CRAG, then score the top one.
    crag = CRAGPipeline()
    crag_result = crag.run(query)
    if not crag_result.relevant_docs:
        print("No relevant candidate to score. Try another query.")
        return

    candidate_id = crag_result.relevant_docs[0].metadata["candidate_id"]
    print(f"Scoring candidate_id={candidate_id} for query: {query!r}")

    selfrag = SelfRAGPipeline()
    show(selfrag.run(query, candidate_id))

    # ── Grounding unit test: a fabricated quote must be caught ───────────────
    print(f"\n{'═'*60}\nGrounding check unit test\n{'═'*60}")
    fake = ScorePoint(
        type="strength",
        claim="Built a quantum teleportation pipeline",
        evidence="Led the quantum teleportation team at NASA in 2099",
    )
    from resume_rag.ingestion.candidate_store import load_candidate_text
    resume = load_candidate_text(candidate_id)
    grounded = verifier.is_grounded(fake, resume)
    print(f"  fabricated quote grounded? {grounded}  (expected: False)")
    assert grounded is False, "verifier failed to catch a fabricated quote!"
    print("  ✓ verifier correctly rejected the fabrication")


if __name__ == "__main__":
    main()
