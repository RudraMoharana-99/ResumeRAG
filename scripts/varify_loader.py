"""Verify loader: print filename, metadata, and first 200 chars for every resume."""

from pathlib import Path
from src.config import get_settings
from src.ingestion.loader import load_all_resumes


def main():
    settings = get_settings()
    resumes = load_all_resumes(settings.raw_dir)

    print(f"\nLoaded {len(resumes)} resumes\n{'-'* 80}")
    for r in resumes:
        print(f"source      : {r['source']}")
        print(f"candidate_id: {r['candidate_id']}")
        print(f"file_type   : {r['file_type']}")
        print(f"text_length : {len(r['text'])} chars")
        print(f"preview     : {r['text'][:200]!r}")
        print("─" * 60)

if __name__ == "__main__":
    main()