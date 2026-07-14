"""Run eval cases through the agent and shape results for RAGAS.

RAGAS expects each sample as:
    user_input        : the question
    retrieved_contexts: list[str] — the chunks the system retrieved
    response          : the answer string (we synthesize from ranked output)
    reference         : ground-truth answer

Option A: we flatten the agent's structured ranked output into a prose
"response" by joining grounded claims, so faithfulness can check them.
"""

from __future__ import annotations
from dataclasses import dataclass

from src.components.retriever import HybridRetriever
from src.eval.dataset import EvalCase
from src.graph.builder import build_agent
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class EvalSample:
    user_input: str
    retrieved_contexts: list[str]
    response: str
    reference: str
    # bookkeeping for our own retrieval metrics
    expected_ids: list[str]
    retrieved_ids: list[str]
    status: str


class EvalRunner:
    def __init__(self) -> None:
        self._agent = build_agent()
        self._retriever = HybridRetriever()   # for capturing contexts directly

    def run_case(self, case: EvalCase) -> EvalSample:
        # 1. Retrieve for our own hit-rate bookkeeping (which candidates surfaced).
        parents = self._retriever.retrieve(case.question)
        retrieved_ids = list(dict.fromkeys(p.metadata["candidate_id"] for p in parents))

        # 2. Run the full agent (rank path) for the answer.
        out = self._agent.invoke({"query": case.question})
        status = out.get("status") or "ok"
        ranked = out.get("ranked", [])

        # 3. Contexts for RAGAS = full resume text of the SCORED candidates.
        #    Self-RAG grounds claims against the full resume, so faithfulness
        #    must verify against that same source — not just retrieved chunks.
        contexts = self._scored_candidate_texts(ranked)

        # 4. Response = clean prose of grounded claims (no ID/score scaffolding).
        response = self._synthesize(ranked, status)

        log.info(
            "case=%r status=%s ranked=%d contexts=%d",
            case.question, status, len(ranked), len(contexts),
        )

        return EvalSample(
            user_input=case.question,
            retrieved_contexts=contexts,
            response=response,
            reference=case.ground_truth,
            expected_ids=case.expected_candidate_ids,
            retrieved_ids=retrieved_ids,
            status=status,
        )

    @staticmethod
    def _scored_candidate_texts(ranked: list) -> list[str]:
        """Full resume text of each scored candidate — the grounding source."""
        from src.ingestion.candidate_store import load_candidate_text
        texts = []
        for s in ranked:
            t = load_candidate_text(s.candidate_id)
            if t:
                texts.append(t)
        return texts or [""]   # RAGAS dislikes empty context lists

    @staticmethod
    def _synthesize(ranked: list, status: str) -> str:
        """Plain claim sentences for RAGAS — no candidate IDs or scores."""
        if status in {"no_strong_match", "not_found"} or not ranked:
            return "No strong candidate matches were found for this query."

        claims = []
        for s in ranked:
            for p in s.grounded_points:
                if p.type == "strength":
                    claims.append(p.claim.rstrip(".") + ".")
        return " ".join(claims) if claims else "No grounded strengths were found."

    def run_all(self, cases: list[EvalCase]) -> list[EvalSample]:
        return [self.run_case(c) for c in cases]