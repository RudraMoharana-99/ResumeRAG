"""Resume file loader.

Supports .pdf(pypdf) and .docx(docx2txt).
Returns a dict with raw text + metadata for each file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import docx2txt
from pypdf import PdfReader


def _candidate_id(path: Path) -> str:
    """Stable 8-char ID derived from filenae (not content - content may vary)."""
    return hashlib.md5(path.name.encode()).hexdigest()[:8]


def load_resume(path: Path) -> dict:
    """Load a single resume file.
    
    Returns:{
        "text": str,
        "candidate_id": str,
        "source": str,
        "file_type": str, 
    }
    
    Raises:
        ValueError: Unsupported file type
        RuntimeError: Parse failure
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _load_pdf(path)
        file_type = "pdf"
    elif suffix == ".docx":
        text = _load_docx(path)
        file_type = "docx"
    else:
        raise ValueError(f"Unsupported file tyoe: {suffix} ({path.name})")

    return {
        "text": text.strip(),
        "candidate_id": _candidate_id(path=path),
        "source": path.name,
        "file_type": file_type
    }

def load_all_resumes(directory: Path) -> list[dict]:
    """Load all .pdf and .docx files from a directory (non-recursive)."""
    directory = Path(directory)
    files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in {".pdf", ".docx"}
    )
    results = []
    for f in files:
        try:
            results.append(load_resume(f))
        except Exception as e:
            print(f"[Warn] skipping {f.name}: {e}")

    return results


#-------private parsers----------------------------------------

def _load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF parse failed for {path.name}: {e}") from e

def _load_docx(path: Path) -> str:
    try:
        return docx2txt.process(str(path))
    except Exception as e:
        raise RuntimeError(f"DOCX parse failed for {path.name}: {e}") from e