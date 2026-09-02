
import json
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

CASES_FILE = Path(__file__).resolve().parent / "test_case3.json"
RESULTS_DIR = Path(__file__).resolve().parent / "test_case_3_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC_DIR))

MAX_RETRIEVAL_CHUNKS = 10


# ============================================================
# CASE LOADING
# ============================================================

def load_cases():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_source(source):
    """
    Make .mp3, .json and other episode filename formats comparable.
    """
    if not source:
        return None

    return Path(str(source)).stem.lower().strip()


def overlaps(a_start, a_end, b_start, b_end):
    """
    Return True when two timestamp ranges overlap.
    """
    try:
        return (
            float(a_start) < float(b_end)
            and float(a_end) > float(b_start)
        )
    except (TypeError, ValueError):
        return False


# ============================================================
# RETRIEVED EVIDENCE
# ============================================================

def extract_evidence(result):
    """
    Extract the top-K retrieved chunks from the LangGraph result.
    """

    evidence = []

    chunks = result.get("retrieved_chunks", [])

    for rank, chunk in enumerate(
        chunks[:MAX_RETRIEVAL_CHUNKS],
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

        timestamp_start = (
            metadata.get("timestamp_start")
            if metadata.get("timestamp_start") is not None
            else metadata.get("start")
        )

        timestamp_end = (
            metadata.get("timestamp_end")
            if metadata.get("timestamp_end") is not None
            else metadata.get("end")
        )

        evidence.append({
            "rank": rank,
            "text": chunk.get("text", ""),
            "episode": episode,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "rerank_score": chunk.get("rerank_score")
        })

    return evidence


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(case, evidence):

    # --------------------------------------------------------
    # Required sources
    # --------------------------------------------------------

    required_sources = {
        normalize_source(source)
        for source in case.get("required_sources", [])
        if source
    }

    retrieved_sources = {
        normalize_source(item["episode"])
        for item in evidence
        if item.get("episode")
    }

    source_coverage = (
        len(required_sources & retrieved_sources)
        / len(required_sources)
        if required_sources
        else 0
    )

    # --------------------------------------------------------
    # Expected evidence
    # --------------------------------------------------------

    expected_evidence = case.get(
        "expected_evidence",
        []
    )

    matched_evidence = 0

    for expected_item in expected_evidence:

        expected_source = normalize_source(
            expected_item.get("source")
        )

        found = False

        for retrieved_item in evidence:

            retrieved_source = normalize_source(
                retrieved_item.get("episode")
            )

            if retrieved_source != expected_source:
                continue

            if overlaps(
                retrieved_item.get("timestamp_start"),
                retrieved_item.get("timestamp_end"),
                expected_item.get("timestamp_start"),
                expected_item.get("timestamp_end")
            ):
                found = True
                break

        if found:
            matched_evidence += 1

    # --------------------------------------------------------
    # Evidence Recall@10
    # --------------------------------------------------------

    evidence_recall = (
        matched_evidence / len(expected_evidence)
        if expected_evidence
        else 0
    )

    # --------------------------------------------------------
    # Precision@10
    #
    # A retrieved chunk is considered relevant if it overlaps
    # with at least one manually annotated expected evidence span.
    # --------------------------------------------------------

    relevant_retrieved = 0

    for retrieved_item in evidence:

        retrieved_source = normalize_source(
            retrieved_item.get("episode")
        )

        if not retrieved_source:
            continue

        for expected_item in expected_evidence:

            expected_source = normalize_source(
                expected_item.get("source")
            )

            if retrieved_source != expected_source:
                continue

            if overlaps(
                retrieved_item.get("timestamp_start"),
                retrieved_item.get("timestamp_end"),
                expected_item.get("timestamp_start"),
                expected_item.get("timestamp_end")
            ):
                relevant_retrieved += 1
                break

    precision_at_10 = (
        relevant_retrieved / len(evidence)
        if evidence
        else 0
    )

    return {
        "source_coverage": round(
            source_coverage,
            3
        ),

        "evidence_recall": round(
            evidence_recall,
            3
        ),

        "precision_at_10": round(
            precision_at_10,
            3
        ),

        "matched_evidence": matched_evidence,

        "expected_evidence": len(
            expected_evidence
        ),

        "required_sources_found": len(
            required_sources & retrieved_sources
        ),

        "required_sources": len(
            required_sources
        )
    }


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(app_graph, case, case_number):

    case_id = case.get(
        "id",
        f"case_{case_number:02d}"
    )

    category = case.get(
        "category",
        "unknown"
    )

    question = case["question"]

    thread_id = (
        f"eval_{case_id}_{uuid.uuid4().hex[:8]}"
    )

    start = time.perf_counter()

    try:

        result = app_graph.invoke(
            {
                "question": question
            },
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        latency = (
            time.perf_counter()
            - start
        )

        evidence = extract_evidence(
            result
        )

        metrics = evaluate_retrieval(
            case,
            evidence
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print(
            f"{case_id.upper()} | {category}"
        )
        print("=" * 70)

        print(
            f"\nQuestion:\n{question}"
        )

        print(
            f"\nAnswer:\n"
            f"{result.get('answer', '')}"
        )

        print(
            f"\nSource coverage: "
            f"{metrics['source_coverage']:.2f}"
        )

        print(
            f"Evidence recall@10: "
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

            "category": category,

            "question": question,

            "rewritten_query": result.get(
                "rewritten_query"
            ),

            "answer": result.get(
                "answer",
                ""
            ),

            "retrieval": {

                "metrics": metrics,

                "evidence": evidence

            },

            "latency_seconds": round(
                latency,
                3
            ),

            "status": "success"
        }

    except Exception as e:

        print(
            f"\n{case_id.upper()} ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return {

            "case_id": case_id,

            "category": category,

            "question": question,

            "status": "error",

            "error": {

                "type": type(e).__name__,

                "message": str(e)

            }
        }


# ============================================================
# AGGREGATE METRICS
# ============================================================

def calculate_summary(results):

    successful = [
        result
        for result in results
        if result["status"] == "success"
    ]

    errors = (
        len(results)
        - len(successful)
    )

    if not successful:

        return {
            "total_cases": len(results),
            "completed": 0,
            "errors": errors,
            "retrieval": {
                "source_coverage": 0,
                "evidence_recall": 0,
                "precision_at_10": 0
            },
            "average_latency_seconds": 0,
            "by_category": {}
        }

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    source_coverage = sum(
        result["retrieval"]["metrics"][
            "source_coverage"
        ]
        for result in successful
    ) / len(successful)

    evidence_recall = sum(
        result["retrieval"]["metrics"][
            "evidence_recall"
        ]
        for result in successful
    ) / len(successful)

    precision_at_10 = sum(
        result["retrieval"]["metrics"][
            "precision_at_10"
        ]
        for result in successful
    ) / len(successful)

    average_latency = sum(
        result["latency_seconds"]
        for result in successful
    ) / len(successful)

    # --------------------------------------------------------
    # Category-level metrics
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for result in successful:

        grouped[
            result.get(
                "category",
                "unknown"
            )
        ].append(result)

    by_category = {}

    for category, category_results in grouped.items():

        category_source_coverage = sum(
            result["retrieval"]["metrics"][
                "source_coverage"
            ]
            for result in category_results
        ) / len(category_results)

        category_evidence_recall = sum(
            result["retrieval"]["metrics"][
                "evidence_recall"
            ]
            for result in category_results
        ) / len(category_results)

        category_precision = sum(
            result["retrieval"]["metrics"][
                "precision_at_10"
            ]
            for result in category_results
        ) / len(category_results)

        category_latency = sum(
            result["latency_seconds"]
            for result in category_results
        ) / len(category_results)

        by_category[category] = {

            "cases": len(
                category_results
            ),

            "source_coverage": round(
                category_source_coverage,
                3
            ),

            "evidence_recall": round(
                category_evidence_recall,
                3
            ),

            "precision_at_10": round(
                category_precision,
                3
            ),

            "average_latency_seconds": round(
                category_latency,
                3
            )
        }

    return {

        "total_cases": len(results),

        "completed": len(
            successful
        ),

        "errors": errors,

        "retrieval": {

            "source_coverage": round(
                source_coverage,
                3
            ),

            "evidence_recall": round(
                evidence_recall,
                3
            ),

            "precision_at_10": round(
                precision_at_10,
                3
            )
        },

        "average_latency_seconds": round(
            average_latency,
            3
        ),

        "by_category": by_category
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nLoading evaluation cases..."
    )

    cases = load_cases()

    print(
        f"Loaded {len(cases)} cases."
    )

    from graph import app_graph

    results = []

    for index, case in enumerate(
        cases,
        start=1
    ):

        results.append(
            run_case(
                app_graph,
                case,
                index
            )
        )

    summary = calculate_summary(
        results
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    timestamp = datetime.now()

    output = {

        "evaluation": {

            "name": "fermi_podcast_companion",

            "timestamp": timestamp.isoformat(),

            "num_cases": len(results),

            "max_retrieval_chunks":
                MAX_RETRIEVAL_CHUNKS

        },

        "summary": summary,

        # Preserve raw outputs for inspection
        "results": results
    }

    filename = (
        RESULTS_DIR
        / (
            f"evaluation_"
            f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
            f".json"
        )
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nCases: "
        f"{summary['total_cases']}"
    )

    print(
        f"Completed: "
        f"{summary['completed']}"
    )

    print(
        f"Errors: "
        f"{summary['errors']}"
    )

    print(
        f"\nSource coverage: "
        f"{summary['retrieval']['source_coverage']:.2f}"
    )

    print(
        f"Evidence recall@10: "
        f"{summary['retrieval']['evidence_recall']:.2f}"
    )

    print(
        f"Precision@10: "
        f"{summary['retrieval']['precision_at_10']:.2f}"
    )

    print(
        f"Average latency: "
        f"{summary['average_latency_seconds']:.2f}s"
    )

    # --------------------------------------------------------
    # Category breakdown
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("BY CATEGORY")
    print("-" * 70)

    for category, metrics in (
        summary["by_category"].items()
    ):

        print(
            f"\n{category}"
        )

        print(
            f"  Cases: "
            f"{metrics['cases']}"
        )

        print(
            f"  Source coverage: "
            f"{metrics['source_coverage']:.2f}"
        )

        print(
            f"  Evidence recall@10: "
            f"{metrics['evidence_recall']:.2f}"
        )

        print(
            f"  Precision@10: "
            f"{metrics['precision_at_10']:.2f}"
        )

        print(
            f"  Average latency: "
            f"{metrics['average_latency_seconds']:.2f}s"
        )

    print(
        f"\nRaw results saved to:\n{filename}"
    )


if __name__ == "__main__":
    main()

