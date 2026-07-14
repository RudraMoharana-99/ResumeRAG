"""Vector store + parent docstore + parent-child retriever factories.

Pattern (LangChain MultiVectorRetriever):
- children -> Chroma (embedded, searched by similarity)
- parents  -> LocalFileStore (persisted to disk, fetched by parent_id)

When the retriever runs, it searches children, looks up each child's parent_id,
and returns the parent Documents to the caller.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_chroma import Chroma

from src.components.embeddings import get_embeddings
from src.config import get_settings

# Metadata key on children that points to their parents docstore key
PARENT_ID_KEY = "parent_id"

@lru_cache(maxsize=1)
def get_child_vectorstore() -> Chroma:
    """Chroma collection holding embedded child chunks."""
    settings = get_settings()
    return Chroma(
        collection_name="resume_children",
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
    )

@lru_cache(maxsize=1)
def get_parent_docstore():
    """Disk-backed key-value store for parent Documents.
    
    LocalFileStore stores bytes; create_kv_docstore wraps it so we can
    store/retrieve LangChain Document objects directly.
    """
    settings = get_settings()
    parent_dir = settings.chroma_persist_dir.parent/"parent_docstore"
    parent_dir.mkdir(parents=True, exist_ok=True)
    fs = LocalFileStore(str(parent_dir))
    return create_kv_docstore(fs)


def get_parent_child_retriever(k: int = 5) -> MultiVectorRetriever:
    """Retriever that searches children and returns the parents they belong to."""
    return MultiVectorRetriever(
        vectorstore=get_child_vectorstore(),
        docstore=get_parent_docstore(),
        id_key=PARENT_ID_KEY,
        search_kwargs={"k": k},
    )
