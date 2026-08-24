"""Parent-Child chunker for resume documents.

Strategy:
    - Parent chunk (large) -> what the LLM reads for context
    - Child chunks(small) -> what gets embedded and retrieved

Each chalid carries a parent_id that links back to its parents.

"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



def chunk_resume(
        resume: dict,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
) -> tuple[list[Document], list[Document]]:
    """Splitting a single resume dict into parent + child Documents
    
    Args: 
        resume: output dict from loader.load_resume()
        parent_chun_size: chars per parent chunk
        parent_chunk_overlap: overlap between parent chunks
        child_chunk_size: chars per child chunk
        child_chunk_overlap: overlap between child chunks
    
    Returns: 
        (parents, children) - both are lists of Langchain Documents
    """
    text = resume['text']
    candidate_id = resume['candidate_id']
    base_meta = {
        "candidate_id": candidate_id,
        "source": resume['source'],
        "file_type": resume["file_type"],
    }

    #---------1. Parent Splits ---------------------
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size = parent_chunk_size,
        chunk_overlap = parent_chunk_overlap,
    )
    parent_texts = parent_splitter.split_text(text)

    parents: list[Document] = []
    children: list[Document] = []

    #---------2. Children Splits --------------------
    children_splitter = RecursiveCharacterTextSplitter(
        chunk_size = child_chunk_size,
        chunk_overlap = child_chunk_overlap,
    )

    for p_idx, p_text in enumerate(parent_texts):
        parent_id = f"{candidate_id}_p{p_idx}"

        parent_doc = Document(
            page_content=p_text,
            metadata={**base_meta, "parent_id": parent_id, "chunk_index": p_idx}
        )
        parents.append(parent_doc)

        child_texts = children_splitter.split_text(p_text)
        for c_idx, c_text in enumerate(child_texts):
            child_doc = Document(
                page_content=c_text,
                metadata={
                    **base_meta,
                    "parent_id": parent_id,
                    "chunk_index": c_idx
                },
            )
            children.append(child_doc)
    return parents, children


def chunk_all_resumes(
        resumes: list[dict],
        **kwargs,
) -> tuple[list[Document], list[Document]]:
    """Chunk a list of resume dicts. Kwargs forwarded to chunk_resume."""
    all_parents: list[Document] = []
    all_children: list[Document] = []

    for resume in resumes:
        parents, children = chunk_resume(resume=resume, **kwargs)
        all_parents.extend(parents)
        all_children.extend(children)

    return all_parents, all_children