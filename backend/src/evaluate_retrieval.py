import json
import time
from pathlib import Path

from retrieval import retrieve

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

    return (
        retrieved_start < correct_end
        and retrieved_end > correct_start
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_pipeline():

    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []

    total_time = 0

    print("\n" + "=" * 80)
    print(" EVALUATING: retrvfinal.py")
    print("=" * 80)

    print(f"Chroma collection  : fermi_transcripts")
    print(f"Evaluation dataset : {EVAL_FILE}")
    print(f"Number of questions: {len(dataset)}")

    print("=" * 80)

    for item in dataset:

        question = item["question"]
        correct_chunks = item["correct_chunks"]

        start_time = time.perf_counter()

        retrieved = retrieve(question)

        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        first_correct_rank = None

        for rank, result in enumerate(retrieved, 1):

            for correct in correct_chunks:

                if is_correct(result, correct):
                    first_correct_rank = rank
                    break

            if first_correct_rank is not None:
                break

        # --------------------------------------------------
        # Metrics for this question
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
        "pipeline": "retrvfinal.py",
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr,
        "avg_latency": avg_latency
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    results = evaluate_pipeline()