"""Cohere reranker singleton.

Cross-encoder: scores(query, chunk) pairs together - fixes lossy cosine ranking.
"""

from __future__ import annotations
from functools import lru_cache
from langchain_cohere import CohereRerank
from src.config import get_settings


@lru_cache
def get_reranker() -> CohereRerank:
    settings = get_settings()
    return CohereRerank(
        cohere_api_key=settings.cohere_api_key,
        model=settings.reranker_model,
        top_n=settings.rerank_top_n,
    )
