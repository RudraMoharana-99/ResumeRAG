"""Hand-built eval dataset over the resume corpus.

Cases reference candidates by FILENAME (human-readable, convention-free).
The runner resolves each filename to its real candidate_id using the same
hash the ingestion loader uses — so eval IDs always match retrieval IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from resume_rag.ingestion.loader import _candidate_id
from pathlib import Path


@dataclass
class EvalCase:
    question: str
    expected_files: list[str] = field(default_factory=list)   # filenames in data/resumes/
    ground_truth: str = ""
    expect_refusal: bool = False

    @property
    def expected_candidate_ids(self) -> list[str]:
        """Resolve filenames -> real candidate_ids (same hash as the loader)."""
        return [_candidate_id(Path(f)) for f in self.expected_files]


DATASET: list[EvalCase] = [
    EvalCase(
        question="machine learning engineer with PyTorch experience",
        expected_files=["03_Ethan_Brooks.pdf", "28_Chloe_Barnes.pdf"],
        ground_truth=(
            "Ethan Brooks is a Machine Learning Engineer with PyTorch experience. "
            "Chloe Barnes has PyTorch experience as a Computer Vision Engineer."
        ),
    ),
    EvalCase(
        question="DevOps engineer with Kubernetes and Docker experience",
        expected_files=["04_Olivia_Carter.pdf", "12_Charlotte_Perry.pdf", "23_Daniel_Watson.pdf"],
        ground_truth=(
            "Olivia Carter, Charlotte Perry, and Daniel Watson have experience "
            "with containerization and Kubernetes-based environments."
        ),
    ),
    EvalCase(
        question="GenAI engineer with LangGraph and RAG experience",
        expected_files=["22_Ella_Richardson.pdf"],
        ground_truth=(
            "Ella Richardson is a GenAI Engineer with LangGraph, Claude API, "
            "RAG systems, FastAPI, and evaluation experience."
        ),
    ),
    EvalCase(
        question="RPA developer with UiPath experience",
        expected_files=["08_Ava_Mitchell.pdf"],
        ground_truth=(
            "Ava Mitchell is an RPA Developer with UiPath, Power Automate, "
            "Python, and SQL experience."
        ),
    ),
    EvalCase(
        question="computer vision engineer using OpenCV and YOLO",
        expected_files=["28_Chloe_Barnes.pdf"],
        ground_truth=(
            "Chloe Barnes is a Computer Vision Engineer with OpenCV, PyTorch, "
            "YOLO, CNN, and MLOps expertise."
        ),
    ),
    EvalCase(
        question="data scientist with SQL and statistical analysis skills",
        expected_files=["02_Sophia_Reed.pdf"],
        ground_truth=(
            "Sophia Reed is a Data Scientist with Python, SQL, Scikit-learn, "
            "Pandas, and Statistics expertise."
        ),
    ),
    EvalCase(
        question="data engineer experienced with Spark and Kafka",
        expected_files=["05_Liam_Foster.pdf"],
        ground_truth=(
            "Liam Foster is a Data Engineer with Spark, Airflow, Python, SQL, "
            "and Kafka experience."
        ),
    ),
    EvalCase(
        question="backend developer experienced in Redis",
        expected_files=["10_Mia_Collins.pdf"],
        ground_truth=(
            "Mia Collins is a Backend Developer with FastAPI, PostgreSQL, "
            "Redis, and Docker experience."
        ),
    ),
    EvalCase(
        question="network engineer with routing and switching experience",
        expected_files=["13_James_Cooper.pdf"],
        ground_truth=(
            "James Cooper is a Network Engineer with Cisco technologies, "
            "routing, switching, firewalls, and VPNs."
        ),
    ),
    EvalCase(
        question="site reliability engineer with Prometheus and Grafana experience",
        expected_files=["23_Daniel_Watson.pdf"],
        ground_truth=(
            "Daniel Watson is a Site Reliability Engineer with Prometheus, "
            "Grafana, Linux, Kubernetes, and Go experience."
        ),
    ),
    # ── refusal cases ─────────────────────────────────────────────────────────
    EvalCase(
        question="blockchain developer with Solidity experience",
        expected_files=[],
        ground_truth="No candidate in the corpus has blockchain or Solidity experience.",
        expect_refusal=True,
    ),
    EvalCase(
        question="frontend developer experienced in Angular",
        expected_files=[],
        ground_truth="No candidate in the corpus has Angular experience.",
        expect_refusal=True,
    ),
]


def get_dataset() -> list[EvalCase]:
    return DATASET