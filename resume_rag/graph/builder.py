"""Assembles the agentic StateGrap (Layer 3).

BATCH 1: routing skeleton with stub nodes. Confirms the graph wiring,
router, and conditional edges work before real logic goes in (Btach 2)
"""

from __future__ import annotations

from langgraph.graph import START, END, StateGraph

from resume_rag.graph.edges import more_to_score, route_by_intent
from resume_rag.graph.router import classify
from resume_rag.graph.state import AgentState

from resume_rag.logger import get_logger
from resume_rag.config import get_settings
from resume_rag.graph.resolve import resolve_names
from resume_rag.crag.pipeline import CRAGPipeline
from resume_rag.selfrag.pipeline import SelfRAGPipeline
from resume_rag.selfrag.scorer import score_candidate
from resume_rag.ingestion.candidate_store import load_candidate_text

log = get_logger(__name__)


# Built once, reused across invocations
_crag = CRAGPipeline()
_selfrag = SelfRAGPipeline()

# ── Nodes (STUBS in Batch 1) ────────────────────────────────────────────────

def route_node(state: AgentState) -> AgentState:
    intent = classify(state["query"])
    return {"intent": intent.intent, "_names": intent.names}


def retrieve_filter_node(state: AgentState) -> AgentState:
    """Rank path. Constrained mode (names given) skips retrieval."""
    names = state.get("_names", [])

    if names:
        # Constrained rank: score exactly these people, no retrieval.
        resolved, unresolved = resolve_names(names)
        if unresolved:
            return {
                "candidate_ids": [],
                "status": "not_found",
                "message": f"Could not find: {', '.join(unresolved)}",
            }
        return {"candidate_ids": list(resolved.values()), "status": "ok"}

    # Open rank: retrieve -> CRAG.
    result = _crag.run(state["query"])
    if result.status == "no_strong_match":
        return {"candidate_ids": [], "status": "no_strong_match",
                "message": "No strong candidate matches for this query."}

    settings = get_settings()
    pool, seen = [], set()
    for doc in result.relevant_docs:
        cid = doc.metadata["candidate_id"]
        if cid not in seen:
            seen.add(cid)
            pool.append(cid)
        if len(pool) >= settings.agent_rank_pool:
            break
    return {"candidate_ids": pool, "status": "ok"}


def score_node(state: AgentState) -> AgentState:
    """Score ONE not-yet-scored candidate via Self-RAG. Loop drives the rest."""
    scored = list(state.get("scored", []))
    done = {s.candidate_id for s in scored}
    remaining = [c for c in state["candidate_ids"] if c not in done]

    cid = remaining[0]
    verified = _selfrag.run(state["query"], cid)
    scored.append(verified)
    return {"scored": scored}


def rank_node(state: AgentState) -> AgentState:
    scored = state.get("scored", [])
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    return {"ranked": ranked, "status": "ok",
            "message": f"Ranked {len(ranked)} candidate(s)."}


def compare_node(state: AgentState) -> AgentState:
    """Two-person head-to-head: resolve, score both, structured diff."""
    names = state.get("_names", [])
    resolved, unresolved = resolve_names(names)
    if unresolved:
        return {"comparison": None, "status": "not_found",
                "message": f"Could not find: {', '.join(unresolved)}"}

    ids = list(resolved.values())
    a, b = _selfrag.run(state["query"], ids[0]), _selfrag.run(state["query"], ids[1])

    comparison = {
        "a": {"candidate_id": a.candidate_id, "score": a.score,
              "summary": a.summary, "points": a.grounded_points},
        "b": {"candidate_id": b.candidate_id, "score": b.score,
              "summary": b.summary, "points": b.grounded_points},
        "winner": a.candidate_id if a.score >= b.score else b.candidate_id,
    }
    return {"comparison": comparison, "compare_ids": ids, "status": "ok",
            "message": "Comparison complete."}



# ── Build ────────────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)

    g.add_node("route", route_node)
    g.add_node("retrieve_filter", retrieve_filter_node)
    g.add_node("score", score_node)
    g.add_node("rank", rank_node)
    g.add_node("compare", compare_node)

    g.add_edge(START, "route")

    # Branch by intent
    g.add_conditional_edges(
        "route", route_by_intent,
        {"rank": "retrieve_filter", "compare": "compare"}
    )

    # Rank subflow: retrieve_filter -> (score loop) -> rank -> END
    # g.add_edge("retrieve_filter", "score")
    g.add_conditional_edges(
        "retrieve_filter", more_to_score,
        {"score": "score", "rank": "rank"},
    )
    g.add_conditional_edges(
        "score", more_to_score,
        {"score": "score", "rank": "rank"},
    )
    g.add_edge("rank", END)

    # Compare subflow
    g.add_edge("compare", END)

    return g.compile()

