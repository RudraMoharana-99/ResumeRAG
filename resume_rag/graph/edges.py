"""Conditional routing functions for the graph."""

from __future__ import annotations

from resume_rag.graph.state import AgentState


def route_by_intent(state: AgentState) -> str:
    """Entry branch: send to the rank or copmare subflow."""
    return state['intent']


def more_to_score(state: AgentState) -> str:
    scored_ids = {s.candidate_id for s in state.get("scored", [])}
    remaining = [c for c in state.get("candidate_ids", []) if c not in scored_ids]
    if not remaining:
        return "rank"
    # If retrieve_filter failed (no pool / not_found), don't loop forever.
    if state.get("status") in {"no_strong_match", "not_found"}:
        return "rank"
    return "score"