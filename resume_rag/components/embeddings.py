"""Embedding model singleton.

Uses BAAI/bge-small-env1.5 (local, no API key required).
"""

from __future__ import annotations
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() ->  HuggingFaceEmbeddings:
    """Return cached embedding model instance."""
    return HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )