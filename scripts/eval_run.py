"""Run the full RAGAS eval: agent over dataset -> metrics -> CSV."""

from src.eval.dataset import get_dataset
from src.eval.ragas_eval import run_ragas
from src.eval.runner import EvalRunner


def main():
    cases = get_dataset()
    print(f"Running {len(cases)} eval cases through the agent...\n")

    runner = EvalRunner()
    samples = runner.run_all(cases)

    print(f"\n{'═'*60}\nRAGAS scoring\n{'═'*60}")
    result = run_ragas(samples)

    print(f"\n{'═'*60}\nAggregate scores\n{'═'*60}")
    print(result)

    # ── Our own retrieval hit-rate (independent of RAGAS) ─────────────────────
    print(f"\n{'═'*60}\nRetrieval hit-rate (expected candidate in retrieved set)\n{'═'*60}")
    hits = 0
    scored = 0
    for case, sample in zip(cases, samples):
        if case.expect_refusal:
            continue
        scored += 1
        found = set(case.expected_candidate_ids) & set(sample.retrieved_ids)
        mark = "✓" if found else "✗"
        print(f"  {mark} {case.question[:50]:<50} {len(found)}/{len(case.expected_candidate_ids)}")
        if found:
            hits += 1
    print(f"\n  hit-rate: {hits}/{scored} cases retrieved >=1 expected candidate")


if __name__ == "__main__":
    main()