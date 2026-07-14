"""Verify CRAG: a strong-match query and a weak-match query (honest refusal)."""

from src.crag.pipeline import CRAGPipeline


def show(title: str, result):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"  status     : {result.status}")
    print(f"  rewritten  : {result.rewritten}")
    print(f"  query_used : {result.query_used!r}")
    print(f"  relevant   : {len(result.relevant_docs)} docs")
    print(f"\n  grades:")

    for doc, grade in result.graded:
        mark = "✓" if grade.relevant else "✗"
        print(f"    {mark} {doc.metadata['source']:<28} {grade.reason}")

def main():
    crag = CRAGPipeline()

    # Strong: corpus has data/SQL/analyst resumes
    show("Query A (expected strong_match): 'data analyst with SQL'",
         crag.run("data analyst with SQL experience"))

    # Weak: corpus has little/no Python-ML — expect rewrite then no_strong_match
    show("Query B (expected no_strong_match): 'python deep learning engineer'",
         crag.run("python deep learning engineer with PyTorch experience"))


if __name__ == "__main__":
    main()