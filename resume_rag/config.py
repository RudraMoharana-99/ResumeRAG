"""Typed configuration loaded once from environment / .env file.

Import `get_settings()` anywhere; it is cached so the .env is parsed only once
and missing required keys fail fast at first call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = repo root (this file lives at src/config.py)

ROOT_DIR = Path(__file__).resolve().parent.parent



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Required secrets (fail fast if absent) ----
    anthropic_api_key: str
    cohere_api_key: str
    
    # ----Models----[Rudra:04-06-206]
    llm_model:str = "claude-sonnet-4-20250514"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "rerank-english-v3.0"  # COHERE

    #-----Vector Store ----[Rudra:04-06-206]
    chroma_persist_dir: Path = ROOT_DIR / "chroma_db"
    chroma_collection: str = "Resume_screening"

    #-----Data Paths-----------
    raw_dir: Path = ROOT_DIR / "data" / "resumes"

    # --- Retrieval defaults ----------
    parent_chunk_size: int = 1000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 200
    child_chunk_overlap: int = 20
    retrieval_k: int = 20  # candidates before rerank
    rerank_top_n: int = 5  # kept after rerank

    log_level: str = "INFO"

    # ---- Cache ----
    cache_enabled: bool = True
    cache_backend: str = "memory"          # "memory" | "valkey" (future)
    cache_threshold: float = 0.95          # cosine similarity for HIT
    cache_ttl_days: int = 20

    # ----- Fallback RRF---------
    rrf_k: int = 60

    # ---- CRAG ----
    crag_min_relevant: int = 1   # min relevant docs to count as a strong match

    # ---- Self-RAG ----
    selfrag_max_retries: int = 1   # initial attempt + this many regenerations

    # ---- Agent ----
    agent_rank_pool: int = 5   # how many CRAG-approved candidates to score for ranking

    # ---- Eval / LangSmith ----
    langsmith_api_key: str = ""
    langsmith_project: str = "resume-rag"
    langsmith_tracing: bool = False
    eval_results_dir: Path = ROOT_DIR / "eval_results"

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type:ignore[call-arg]