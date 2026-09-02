# Retrieval Evaluation & Improvement

## 1. Objective

The goal of this evaluation was to measure retrieval quality for the Fermi Podcast Companion and select a retrieval architecture based on:

* Retrieval accuracy
* Ranking quality
* Latency
* Context quality

Four retrieval implementations were evaluated using the same benchmark and timestamp-overlap correctness criterion.

---

## 2. Evaluation Setup

### Dataset

* **20 manually curated questions**
* Each question has one or more ground-truth timestamp ranges.
* Ground truth was created from the supplied podcast transcripts.
* A retrieved chunk is considered relevant when its timestamp interval overlaps a ground-truth interval.

### Metrics

**Recall@5**
Percentage of questions with at least one relevant chunk in the top 5 results.

**Recall@10**
Percentage of questions with at least one relevant chunk in the top 10 results.

**MRR (Mean Reciprocal Rank)**
Measures how highly the first relevant result is ranked.

**Average Retrieval Latency**
Average time required to retrieve and rank results for one query.

### Evaluation Command

```bash
python src/eval_all.py
```

The same dataset, correctness criterion, and metrics were used across all four implementations.

---

# 3. Retrieval Pipelines

## V1 — `retrieve.py`

Baseline vector retrieval.

```text
Query
  ↓
Embedding
  ↓
ChromaDB Vector Search
  ↓
Top-10 Chunks
```

Configuration:

* Flat 300-token chunks
* 50-token overlap
* `all-MiniLM-L6-v2`
* ChromaDB
* No reranking

---

## V2 — `retrieve2.py`

Parent-child retrieval with CrossEncoder reranking.

```text
Query
  ↓
Embedding
  ↓
Vector Search
  ↓
Top-30 Child Candidates
  ↓
CrossEncoder Reranking
  ↓
Final Results
```

Configuration:

* 600-token parent chunks
* 100-token child chunks
* 20-token child overlap
* Child chunks embedded for retrieval
* CrossEncoder reranking

---

## V3 — `retrieve2_improved.py`

Parent-child retrieval with parent context supplied to the reranker.

```text
Query
  ↓
Vector Search
  ↓
Child Candidates
  ↓
Child + Parent Context
  ↓
CrossEncoder
  ↓
Reranked Results
```

The purpose of this experiment was to determine whether providing broader parent context would improve ranking quality.

---

## Final — `retrvfinal.py`

Flat-chunk retrieval with CrossEncoder reranking.

```text
Query
  ↓
Embedding
  ↓
Top-30 Vector Candidates
  ↓
CrossEncoder Reranking
  ↓
Top-5 Results
```

Configuration:

* 300-token flat chunks
* 50-token overlap
* `all-MiniLM-L6-v2`
* ChromaDB
* Top-30 vector candidates
* `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Final Top-5 results

---

# 4. Results

| Pipeline                | Recall@5 | Recall@10 |       MRR | Avg Latency |
| ----------------------- | -------: | --------: | --------: | ----------: |
| `retrieve.py`           |      70% |      100% |     0.604 |      0.097s |
| `retrieve2.py`          |      85% |       95% |     0.673 |      0.556s |
| `retrieve2_improved.py` |      90% |       90% | **0.735** |      2.549s |
| `retrvfinal.py`         |  **95%** |  **100%** |     0.707 |      1.993s |

### Key observations

Adding CrossEncoder reranking to the parent-child architecture improved Recall@5 from 70% to 85%.

Adding parent context increased Recall@5 further to 90% and produced the highest MRR of 0.735. However, it also increased latency substantially.

The final flat-chunk + CrossEncoder architecture achieved the best Recall@5 at **95%** and recovered **100% Recall@10**.

---

# 5. Improvement Over Baseline

The final system improved Recall@5:

```text
70% → 95%
```

This is a **25 percentage-point improvement**.

MRR also improved:

```text
0.604 → 0.707
```

The main trade-off was retrieval latency:

```text
0.097s → 1.993s
```

This increase is primarily due to CrossEncoder reranking of the top-30 vector candidates.

---

# 6. Failure Analysis — Q8

### Question

> What is the Unruh effect and how does it relate to black hole radiation?

| Pipeline                | First Correct Rank |
| ----------------------- | -----------------: |
| `retrieve.py`           |                  7 |
| `retrieve2.py`          |      **Not found** |
| `retrieve2_improved.py` |      **Not found** |
| `retrvfinal.py`         |                  6 |

Q8 is useful because the explanation is distributed across multiple parts of the transcript rather than appearing as one isolated statement:

```text
Unruh effect
    ↓
Acceleration produces perceived temperature
    ↓
Equivalence between acceleration and gravity
    ↓
Gravitational field near a horizon
    ↓
Connection to Hawking radiation
```

Both parent-child implementations failed to retrieve a ground-truth chunk within the evaluated top-10 results.

The final flat-chunk + CrossEncoder system recovered relevant evidence at rank 6.

### Interpretation

This failure was important in the final architecture decision. Adding hierarchical parent context did not consistently improve retrieval of multi-step conceptual explanations.

---

# 7. Why Parent-Child Retrieval Was Not Selected

Parent-child retrieval was tested because smaller child chunks can provide more precise matching while parent chunks provide broader context.

However, the benchmark showed:

* **V2:** 85% Recall@5
* **V3:** 90% Recall@5
* **Final:** **95% Recall@5**

V3 achieved the highest MRR, but also had the highest latency at **2.549s/query** and still failed Q8.

The final flat-chunk architecture achieved both:

* **95% Recall@5**
* **100% Recall@10**

Therefore, parent-child chunking was not used in the final pipeline.

The decision was based on measured performance rather than assuming that a more hierarchical architecture would be better.

> **For this corpus and evaluation set, parent-child retrieval added complexity and latency without providing enough retrieval improvement to justify its use.**

This does not imply that parent-child retrieval is generally inferior. It means that it was not the best trade-off for this specific corpus and benchmark.

---

# 8. Final Retrieval Architecture

The selected architecture is:

```text
Podcast Audio
      ↓
Transcription
      ↓
300-token chunks + 50-token overlap
      ↓
MiniLM Embeddings
      ↓
ChromaDB
      ↓
Top-30 Vector Candidates
      ↓
CrossEncoder Reranking
      ↓
Top-5 Evidence Chunks
      ↓
LLM
      ↓
Grounded Answer + Timestamp References
```

### Final configuration

| Component           | Selected Approach                      |
| ------------------- | -------------------------------------- |
| Chunking            | 300-token flat chunks                  |
| Overlap             | 50 tokens                              |
| Embedding           | `all-MiniLM-L6-v2`                     |
| Vector DB           | ChromaDB                               |
| Candidate retrieval | Top 30                                 |
| Reranker            | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Final context       | Top 5                                  |
| Primary metric      | Recall@5                               |

The final architecture was selected using benchmark performance, failure analysis, and latency trade-offs.

---

# 9. End-to-End Evaluation

Retrieval evaluation was complemented by an end-to-end evaluation of the conversational system.

The evaluation contained **13 distinct test cases**, executed repeatedly across **9 evaluation runs**.

The 13 cases covered:

* Deep single-source questions
* Multi-source questions
* Precise factual questions

The repeated runs were used to compare retrieval strategies and assess consistency. Therefore:

> **41 executions does not mean 41 unique evaluation cases. The evaluation contains 13 unique cases and 41 total executions across repeated runs.**

### End-to-end findings

The main weaknesses identified were:

1. **Low retrieval precision** — many retrieved chunks were not directly relevant.
2. **Incomplete evidence recall** — some questions required multiple evidence pieces that were not all retrieved.
3. **Multi-source degradation** — questions requiring information from multiple episodes were harder to retrieve reliably.
4. **Service-level failure** — one run encountered an LLM service error even though retrieval itself succeeded.

The evaluation also confirmed that the system generally provides timestamped answers when sufficient podcast evidence is retrieved and refuses unsupported questions rather than relying on unsupported model knowledge.

---

# 10. Conversation Checks

A separate conversational smoke test checked:

| Scenario                   | Result                                        |
| -------------------------- | --------------------------------------------- |
| Follow-up question         | Correctly resolved using conversation history |
| Supported podcast question | Answered with transcript timestamp            |
| Unsupported question       | Correctly refused                             |
| Conversational context     | Correctly retained                            |

The system is intentionally grounded in the supplied podcast evidence. Unsupported questions should result in a clear refusal rather than an answer based solely on the model's general knowledge.

### Product limitation

Strict grounding can reduce usefulness for general educational questions.

For example, the system correctly refused:

> "Can you give me an example of a black hole?"

While this is safer from a hallucination perspective, it is less helpful when the user is asking for general educational context.

A future improvement would distinguish between:

* **Podcast-grounded questions** → answer using supplied evidence and timestamps.
* **General educational questions** → provide clearly labeled outside knowledge.

This would preserve source trust while making the conversational experience more useful.

---

# 11. Conclusion

The retrieval system was developed through an iterative evaluation process:

```text
Baseline
   ↓
Measure retrieval weaknesses
   ↓
Add CrossEncoder reranking
   ↓
Experiment with parent-child retrieval
   ↓
Analyze failures and latency
   ↓
Select the highest-performing architecture
```

The final system achieved:

* **95% Recall@5**
* **100% Recall@10**
* **0.707 MRR**

compared with the baseline:

* **70% Recall@5**
* **100% Recall@10**
* **0.604 MRR**

The main engineering conclusion is:

> **A flat-chunk architecture combined with candidate generation and CrossEncoder reranking provided the best measured retrieval performance for this corpus.**

The experiments also demonstrated that increased architectural complexity does not automatically produce better retrieval. Parent-child retrieval improved some ranking metrics, but its additional complexity and latency were not justified by the overall benchmark results.

The final architecture was therefore selected based on **reproducible measurements, failure analysis, and explicit performance trade-offs**, rather than architectural complexity alone.
