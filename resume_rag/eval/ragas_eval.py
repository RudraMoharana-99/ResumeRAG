"""RAGAS evaluation: faithfulness, context precision, context recall.

Maps our EvalSamples ito a RAGAS EvaluationDataset and scores them.
Saves pre-question + aggregate results to CSV (the before/after artifact).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from resume_rag._compat import install_vertexai_shim
install_vertexai_shim()
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from resume_rag.components.embeddings import get_embeddings
from resume_rag.components.llm import get_llm
from resume_rag.config import get_settings
from resume_rag.eval.runner import EvalSample
from resume_rag.logger import get_logger

log = get_logger(__name__)


def _to_ragas_dataset(samples: list[EvalSample]) -> EvaluationDataset:
    """Convert EvalSamples into RAGAS's expected schema."""
    records = []
    for s in samples:
        records.append({
            "user_input": s.user_input,
            "retrieved_contexts": s.retrieved_contexts or [""],  # RAGAS dislikes empty
            "response": s.response,
            "reference": s.reference,
        })
    return EvaluationDataset.from_list(records)


def run_ragas(samples: list[EvalSample]) -> dict:
    """Score samples with RAGAS. Returns the aggregate result dict."""
    settings = get_settings()

    # RAGAS needs an LLM + embeddings for its judge metrics. Reuse ours.
    judge_llm = LangchainLLMWrapper(get_llm())
    judge_emb = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(llm=judge_llm),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm)
    ]

    dataset = _to_ragas_dataset(samples=samples)
    log.info("Running RAGAS on %d samples...", len(samples))
    result = evaluate(dataset=dataset, metrics=metrics)

    _save_csv(samples, result, settings.eval_results_dir)

    return result

def _save_csv(samples: list[EvalSample], result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"ragas_{stamp}.csv"

    df = result.to_pandas()
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("RAGAS results saved: %s", path)