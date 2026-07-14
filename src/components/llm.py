"""LLM singleton (Claude via langchain-anthropic).

temprature=0 - graders and scorers need determinism. A separate higher-temp
factory can be added later if generation needs variety.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from src.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=4096,
    )



