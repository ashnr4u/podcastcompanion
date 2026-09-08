import json
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

CASES_FILE = Path(__file__).resolve().parent / "test_case1.json"
RESULTS_DIR = Path(__file__).resolve().parent / "test_case_3_results_new_eval"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC_DIR))

MAX_RETRIEVAL_CHUNKS = 10


# ============================================================
# LOAD CASES
# ============================================================

def load_cases():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# HELPERS
# ============================================================

def normalize_source(source):
    if not source:
        return None
    return Path(str(source)).stem.lower().strip()


def overlaps(a_start, a_end, b_start, b_end):
    try:
        return (
            float(a_start) < float(b_end)
            and float(a_end) > float(b_start)
        )
    except (TypeError, ValueError):
        return False


# ============================================================
# EXTRACT TOP-K EVIDENCE
# ============================================================

def extract_evidence(result):

    evidence = []

    for rank, chunk in enumerate(
        result.get("retrieved_chunks", [])[:MAX_RETRIEVAL_CHUNKS],
        start=1
    ):
        metadata = chunk.get("metadata", {})

        episode = (
            metadata.get("episode")
            or metadata.get("source")
            or metadata.get("file")
            or metadata.get("title")
            or metadata.get("video_id")
        )

        start = (
            metadata.get("timestamp_start")
            if metadata.get("timestamp_start") is not None
            else metadata.get("start")
        )

        end = (
            metadata.get("timestamp_end")
            if metadata.get("timestamp_end") is not None
            else metadata.get("end")
        )

        evidence.append({
            "rank": rank,
            "text": chunk.get("text", ""),
            "episode": episode,
            "timestamp_start": start,
            "timestamp_end": end,
            "rerank_score": chunk.get("rerank_score")
        })

    return evidence


# ============================================================
# EVALUATE RETRIEVAL
# ============================================================

def evaluate_retrieval(case, evidence):

    required_sources = {
        normalize_source(source)
        for source in case.get("required_sources", [])
        if source
    }

    retrieved_sources = {
        normalize_source(item.get("episode"))
        for item in evidence
        if item.get("episode")
    }

    source_coverage = (
        len(required_sources & retrieved_sources)
        / len(required_sources)
        if required_sources else 0
    )

    expected_evidence = case.get("expected_evidence", [])

    # Evidence Recall@10
    matched_evidence = 0

    for expected in expected_evidence:

        expected_source = normalize_source(
            expected.get("source")
        )

        for retrieved in evidence:

            if normalize_source(
                retrieved.get("episode")
            ) != expected_source:
                continue

            if overlaps(
                retrieved.get("timestamp_start"),
                retrieved.get("timestamp_end"),
                expected.get("timestamp_start"),
                expected.get("timestamp_end")
            ):
                matched_evidence += 1
                break

    evidence_recall = (
        matched_evidence / len(expected_evidence)
        if expected_evidence else 0
    )

    # Precision@10
    relevant_retrieved = 0

    for retrieved in evidence:

        retrieved_source = normalize_source(
            retrieved.get("episode")
        )

        for expected in expected_evidence:

            if retrieved_source != normalize_source(
                expected.get("source")
            ):
                continue

            if overlaps(
                retrieved.get("timestamp_start"),
                retrieved.get("timestamp_end"),
                expected.get("timestamp_start"),
                expected.get("timestamp_end")
            ):
                relevant_retrieved += 1
                break

    precision_at_10 = (
        relevant_retrieved / len(evidence)
        if evidence else 0
    )

    return {
        "source_coverage": round(source_coverage, 3),
        "evidence_recall": round(evidence_recall, 3),
        "precision_at_10": round(precision_at_10, 3)
    }


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(app_graph, case, case_number):

    case_id = case.get(
        "id",
        f"case_{case_number:02d}"
    )

    question = case["question"]

    thread_id = (
        f"eval_{case_id}_{uuid.uuid4().hex[:8]}"
    )

    start_time = time.perf_counter()

    try:

        result = app_graph.invoke(
            {"question": question},
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        latency = time.perf_counter() - start_time

        evidence = extract_evidence(result)

        metrics = evaluate_retrieval(
            case,
            evidence
        )

        print("\n" + "=" * 60)
        print(case_id.upper())
        print("=" * 60)

        print(f"\nQuestion:\n{question}")

        print(f"\nAnswer:\n{result.get('answer', '')}")

        print(
            f"\nSource Coverage: "
            f"{metrics['source_coverage']:.2f}"
        )

        print(
            f"Evidence Recall@10: "
            f"{metrics['evidence_recall']:.2f}"
        )

        print(
            f"Precision@10: "
            f"{metrics['precision_at_10']:.2f}"
        )

        print(
            f"Latency: "
            f"{latency:.2f}s"
        )

        return {
            "case_id": case_id,
            "category": case.get("category", "unknown"),
            "question": question,
            "answer": result.get("answer", ""),
            "retrieval": {
                "metrics": metrics,
                "evidence": evidence
            },
            "latency_seconds": round(latency, 3),
            "status": "success"
        }

    except Exception as e:

        print(
            f"\n{case_id.upper()} ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return {
            "case_id": case_id,
            "category": case.get("category", "unknown"),
            "question": question,
            "status": "error",
            "error": {
                "type": type(e).__name__,
                "message": str(e)
            }
        }


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(results):

    successful = [
        r for r in results
        if r["status"] == "success"
    ]

    if not successful:
        return {
            "total_cases": len(results),
            "completed": 0,
            "errors": len(results),
            "retrieval": {
                "source_coverage": 0,
                "evidence_recall": 0,
                "precision_at_10": 0
            },
            "average_latency_seconds": 0
        }

    def average(metric):
        return sum(
            r["retrieval"]["metrics"][metric]
            for r in successful
        ) / len(successful)

    return {
        "total_cases": len(results),
        "completed": len(successful),
        "errors": len(results) - len(successful),
        "retrieval": {
            "source_coverage": round(
                average("source_coverage"), 3
            ),
            "evidence_recall": round(
                average("evidence_recall"), 3
            ),
            "precision_at_10": round(
                average("precision_at_10"), 3
            )
        },
        "average_latency_seconds": round(
            sum(r["latency_seconds"] for r in successful)
            / len(successful),
            3
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    cases = load_cases()

    print(f"\nLoaded {len(cases)} cases.")

    from graph import app_graph

    results = [
        run_case(app_graph, case, i)
        for i, case in enumerate(cases, start=1)
    ]

    summary = calculate_summary(results)

    timestamp = datetime.now()

    output = {
        "evaluation": {
            "name": "fermi_podcast_companion",
            "timestamp": timestamp.isoformat(),
            "num_cases": len(results),
            "max_retrieval_chunks": MAX_RETRIEVAL_CHUNKS
        },
        "summary": summary,
        "results": results
    }

    filename = (
        RESULTS_DIR
        / f"evaluation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nCases: {summary['total_cases']}")
    print(f"Completed: {summary['completed']}")
    print(f"Errors: {summary['errors']}")

    print(
        f"\nSource Coverage: "
        f"{summary['retrieval']['source_coverage']:.2f}"
    )

    print(
        f"Evidence Recall@10: "
        f"{summary['retrieval']['evidence_recall']:.2f}"
    )

    print(
        f"Precision@10: "
        f"{summary['retrieval']['precision_at_10']:.2f}"
    )

    print(
        f"Average Latency: "
        f"{summary['average_latency_seconds']:.2f}s"
    )

    print(f"\nResults saved to:\n{filename}")


if __name__ == "__main__":
    main()