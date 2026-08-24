"""Verify parent-child chunking: counts + one linked pair per resume."""

from resume_rag.config import get_settings
from resume_rag.ingestion.loader import load_all_resumes
from resume_rag.ingestion.chunker import chunk_all_resumes


def main():
    settings = get_settings()
    resumes = load_all_resumes(settings.raw_dir)
    parents, children = chunk_all_resumes(resumes=resumes)

    print(f"\nTotal parents: {len(parents)}")
    print(f"Total children: {len(children)}")
    print(f"{'-'*60}")

    # Print one parent-child pair per unique candidate
    seen = set()
    for parent in parents:
        cid = parent.metadata["candidate_id"]
        if cid in seen:
            continue
        seen.add(cid)

        parent_id = parent.metadata["parent_id"]
        linked_children = [c for c in children if c.metadata['parent_id'] ==parent_id]

        print(f"\ncandidate_id: {cid}")
        print(f"source       : {parent.metadata['source']}")
        print(f"parent_id    : {parent_id}")
        print(f"parent text  : {parent.page_content[:150]!r}")
        print(f"child count  : {len(linked_children)}")
        if linked_children:
            print(f"child[0] text: {linked_children[0].page_content!r}")
        print("─" * 60)


if __name__ == "__main__":
    main()