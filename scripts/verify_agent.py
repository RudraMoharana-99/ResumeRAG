"""Batch 1: verify the graph compiles and routes correctly (stub nodes)."""


from resume_rag.graph.builder import build_agent
from resume_rag.graph.router import classify


# def main():
#     #  ── Router robustness: varied phrasings ──────────────────────────────────
#     print("=" * 60, "\nRouter classification\n" "=" * 60, sep="")

#     for q in [
#         "top candidates for a QC microbiologist role",
#         "who has SQL experience",
#         "compare Manish vs Lopamudra",
#         "rank the best fits for this job description",
#     ]:
#         intent = classify(q)
#         print(f"    {intent.intent:8} names={intent.names}  <-{q!r}")

#         # ── Graph wiring: run both paths with stubs ──────────────────────────────
#         agent = build_agent()

#         print("\n", "═" * 60, "\nRANK path (stub)\n", "═" * 60, sep="")
#         out = agent.invoke({"query": "top candidates for a data analyst role"})
#         print(f"  intent : {out.get('intent')}")
#         print(f"  status : {out.get('status')}")
#         print(f"  message: {out.get('message')}")

#         print("\n", "=" * 60, "\nCOMPARE path (stub)\n", "═" * 60, sep="")
#         out = agent.invoke({"query": "Compare Manish vs Lopamudra"})
#         print(f"    intent   : {out.get('intent')}")
#         print(f"    status   : {out.get('status')}")
#         print(f"    messages : {out.get('message')}")

def show_rank(out):
    print(f"  intent : {out.get('intent')}   status: {out.get('status')}")
    print(f"  message: {out.get('message')}")
    for i, s in enumerate(out.get("ranked", []), 1):
        print(f"\n  #{i}  candidate={s.candidate_id}  score={s.score}/100  "
              f"grounded={s.fully_grounded}")
        print(f"      {s.summary}")
        for p in s.grounded_points[:3]:
            print(f"        [{p.type}] {p.claim}")


def show_compare(out):
    print(f"  intent : {out.get('intent')}   status: {out.get('status')}")
    print(f"  message: {out.get('message')}")
    c = out.get("comparison")
    if not c:
        return
    for side in ("a", "b"):
        d = c[side]
        print(f"\n  {side.upper()}: candidate={d['candidate_id']}  score={d['score']}/100")
        print(f"     {d['summary']}")
    print(f"\n  winner: {c['winner']}")


def main():
    agent = build_agent()

    print("═" * 60, "\n1. OPEN RANK: 'data analyst with SQL experience'\n", "═" * 60, sep="")
    show_rank(agent.invoke({"query": "data analyst with SQL experience"}))

    print("\n", "═" * 60, "\n2. COMPARE: two named candidates\n", "═" * 60, sep="")
    show_compare(agent.invoke({"query": "compare Manish vs Lopamudra for a QC role"}))

    print("\n", "═" * 60, "\n3. NOT FOUND: absent names\n", "═" * 60, sep="")
    show_compare(agent.invoke({"query": "compare Alice vs Bob"}))

    print("\n", "═" * 60, "\n4. NO STRONG MATCH: 'python deep learning engineer'\n", "═" * 60, sep="")
    show_rank(agent.invoke({"query": "python deep learning engineer with PyTorch"}))


if __name__ == "__main__":
    main()