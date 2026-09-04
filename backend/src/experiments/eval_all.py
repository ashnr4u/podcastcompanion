import json
import time
from pathlib import Path

from backend.src.evaluate.retrieve import retrieve_chunks
from backend.src.evaluate.retrieve2 import retrieve
from backend.src.evaluate.retrieve2_improved import retrieve as retrieve_improved
from retrvfinal import retrieve as retrieve_final


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_FILE = BASE_DIR / "data" / "evaluate_retrieve.json"

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)["dataset"]


# --------------------------------------------------
# Correctness Check
# --------------------------------------------------

def is_correct(retrieved, correct):

    retrieved_start = retrieved["metadata"]["start"]
    retrieved_end = retrieved["metadata"]["end"]

    correct_start = correct["start_time"]
    correct_end = correct["end_time"]

    overlap = (
        retrieved_start <= correct_end
        and retrieved_end >= correct_start
    )

    return overlap


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_pipeline(
    name,
    retrieval_function,
    retrieval_file,
    collection
):

    hits_at_5 = 0
    hits_at_10 = 0

    reciprocal_ranks = []

    total_time = 0

    print("\n" + "=" * 80)
    print(f" EVALUATING: {name}")
    print("=" * 80)

    print(f"Retrieval file     : {retrieval_file}")
    print(f"Chroma collection  : {collection}")
    print(f"Evaluation dataset : {EVAL_FILE}")
    print(f"Number of questions: {len(dataset)}")

    print("=" * 80)

    for item in dataset:

        question = item["question"]
        correct_chunks = item["correct_chunks"]

        start_time = time.perf_counter()

        retrieved = retrieval_function(question)

        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        first_correct_rank = None

        # --------------------------------------------------
        # Find first relevant result
        # --------------------------------------------------

        for rank, result in enumerate(retrieved, 1):

            for correct in correct_chunks:

                if is_correct(result, correct):

                    first_correct_rank = rank
                    break

            if first_correct_rank is not None:
                break

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        if first_correct_rank is not None:

            if first_correct_rank <= 5:
                hits_at_5 += 1

            if first_correct_rank <= 10:
                hits_at_10 += 1

            reciprocal_ranks.append(
                1 / first_correct_rank
            )

        else:

            reciprocal_ranks.append(0)

        print(
            f"Q{item['id']:02d} | "
            f"First correct rank: "
            f"{first_correct_rank if first_correct_rank else 'NOT FOUND'} | "
            f"Time: {elapsed:.3f}s"
        )

    # --------------------------------------------------
    # Final Metrics
    # --------------------------------------------------

    total_questions = len(dataset)

    recall_5 = hits_at_5 / total_questions
    recall_10 = hits_at_10 / total_questions
    mrr = sum(reciprocal_ranks) / total_questions
    avg_latency = total_time / total_questions

    print("\n---------- RESULTS ----------")

    print(f"Questions:     {total_questions}")
    print(f"Recall@5:      {recall_5:.3f}")
    print(f"Recall@10:     {recall_10:.3f}")
    print(f"MRR:           {mrr:.3f}")
    print(f"Avg latency:   {avg_latency:.3f}s")

    return {
        "pipeline": name,
        "retrieval_file": retrieval_file,
        "collection": collection,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr,
        "avg_latency": avg_latency
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    results = []

    # ==================================================
    # 1. Baseline - retrieve.py
    # ==================================================

    results.append(
        evaluate_pipeline(
            name="Baseline - retrieve.py",
            retrieval_function=retrieve_chunks,
            retrieval_file="retrieve.py",
            collection="fermi_transcripts"
        )
    )


    # ==================================================
    # 2. retrieve2.py
    # ==================================================

    results.append(
        evaluate_pipeline(
            name="V2 - retrieve2.py",
            retrieval_function=retrieve,
            retrieval_file="retrieve2.py",
            collection="fermi_parent_child"
        )
    )


    # ==================================================
    # 3. retrieve2_improved.py
    # ==================================================

    results.append(
        evaluate_pipeline(
            name="V3 - retrieve2_improved.py",
            retrieval_function=retrieve_improved,
            retrieval_file="retrieve2_improved.py",
            collection="fermi_parent_child"
        )
    )


    # ==================================================
    # 4. Final - retrvfinal.py
    # ==================================================

    results.append(
        evaluate_pipeline(
            name="Final - retrvfinal.py",
            retrieval_function=retrieve_final,
            retrieval_file="retrvfinal.py",
            collection="fermi_transcripts"
        )
    )


    # ==================================================
    # FINAL COMPARISON
    # ==================================================

    print("\n\n")
    print("=" * 100)
    print(" FINAL RETRIEVAL COMPARISON")
    print("=" * 100)

    print(
        f"{'Pipeline':<32}"
        f"{'Recall@5':<12}"
        f"{'Recall@10':<12}"
        f"{'MRR':<10}"
        f"{'Latency':<12}"
    )

    print("-" * 100)

    for result in results:

        print(
            f"{result['pipeline']:<32}"
            f"{result['recall@5']:<12.3f}"
            f"{result['recall@10']:<12.3f}"
            f"{result['mrr']:<10.3f}"
            f"{result['avg_latency']:<12.3f}"
        )

    print("=" * 100)