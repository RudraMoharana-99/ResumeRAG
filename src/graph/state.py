"""Shared state for agentic graph (Layer 3).

LangGraph threads one state dict through every node. Each node returns a 
partial update which LangGraph merges in. Keep it flat and explicit. 
"""

from __future__ import annotations

from typing import TypedDict

from src.selfrag.pipeline import VerifiedScore


class AgentState(TypedDict, total=False):
    query: str
    intent: str                       # 'rank' | 'comapre'

    # rank path
    candidate_ids: list [str]         # pool to score
    scored: list[VerifiedScore]       # accumlates across the score loop
    ranked: list[VerifiedScore]       

    # compare path
    compare_ids: list[str]             # [id_a, id_b]
    comparision: dict | None

    # shared
    ststus: str                         # "ok" | "no_strong_match" | "not found"
    message: str

    _names: list[str]      # carried from router for name resolution