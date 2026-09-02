# Retrieval Evaluation & Improvement

## 1. Objective

The goal of this evaluation was to measure retrieval quality for the Fermi Podcast Companion and determine which retrieval architecture provides the best balance of:

* Retrieval accuracy
* Ranking quality
* Latency
* Context quality

The evaluation compares four retrieval implementations developed during the project.

---

## 2. Evaluation Setup

### Dataset

* **20 manually curated questions**
* Each question has one or more ground-truth timestamp ranges
* Ground truth is based on the supplied podcast transcripts
* A retrieved chunk is considered correct when its timestamp interval overlaps a ground-truth interval

### Metrics

**Recall@5**

Percentage of questions where at least one relevant chunk appears in the top 5 results.

**Recall@10**

Percentage of questions where at least one relevant chunk appears in the top 10 results.

**MRR (Mean Reciprocal Rank)**

Measures how highly the first relevant result is ranked.

**Average Retrieval Latency**

Average time required to retrieve results for one query.

### Evaluation Command

```bash
python src/eval_all.py
```

The same dataset, timestamp-overlap correctness criterion, and metrics are used for all four pipelines.

---

# 3. Retrieval Pipelines Evaluated

## V1 — `retrieve.py`

**Architecture:**

```text
Query
  ↓
Embedding
  ↓
ChromaDB Vector Search
  ↓
Top-10 chunks
```

Configuration:

* Flat transcript chunks
* 300-token chunks
* 50-token overlap
* `all-MiniLM-L6-v2`
* ChromaDB
* No reranking

This was used as the baseline.

---

## V2 — `retrieve2.py`

**Architecture:**

```text
Query
  ↓
Embedding
  ↓
Vector Search
  ↓
Top-30 candidates
  ↓
CrossEncoder Reranking
  ↓
Final results
```

Uses the parent-child collection:

* 600-token parent chunks
* 100-token child chunks
* 20-token child overlap
* Only child chunks are embedded
* CrossEncoder used for reranking

---

## V3 — `retrieve2_improved.py`

This version extends the parent-child approach by providing the CrossEncoder with both the retrieved child and its parent context.

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

The goal was to determine whether additional parent context would improve ranking quality.

---

## Final — `retrvfinal.py`

The final selected architecture returns to flat transcript chunks while retaining CrossEncoder reranking.

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

The baseline vector search achieved 70% Recall@5.

Adding CrossEncoder reranking with parent-child retrieval increased Recall@5 to 85%.

Adding parent context further increased Recall@5 to 90% and produced the highest MRR of 0.735. However, it also increased latency substantially and still failed to retrieve some relevant questions within the top 10.

The final flat-chunk + CrossEncoder approach achieved the best Recall@5 at **95%** while also recovering **100% Recall@10**.

---

# 5. Improvement Over Baseline

The final system improved Recall@5 from:

```text
70% → 95%
```

This is a **25 percentage-point improvement**.

MRR improved from:

```text
0.604 → 0.707
```

The final system therefore provides substantially better top-ranked retrieval than the baseline.

The trade-off is latency:

```text
0.097s → 1.993s
```

The increase is primarily due to CrossEncoder reranking of the vector-retrieved candidate set.

---

# 6. Failure Analysis — Q8

### Question

> What is the Unruh effect and how does it relate to black hole radiation?

### Results

| Pipeline                | First Correct Rank |
| ----------------------- | -----------------: |
| `retrieve.py`           |                  7 |
| `retrieve2.py`          |      **NOT FOUND** |
| `retrieve2_improved.py` |      **NOT FOUND** |
| `retrvfinal.py`         |                  6 |

Q8 is a useful failure case because the answer is not contained in one isolated statement.

The conceptual explanation is distributed across several transcript sections:

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

The two parent-child approaches failed to retrieve a ground-truth chunk within the evaluated top-10 results.

The final flat-chunk + CrossEncoder system recovered the relevant evidence at rank 6.

### Interpretation

This suggests that, for this dataset, adding hierarchical parent context did not consistently improve retrieval of multi-step conceptual explanations.

The parent-child approaches were more complex but did not outperform the final flat-chunk architecture on the primary Recall@5 metric.

---

# 7. Why Parent-Child Retrieval Was Not Selected

Parent-child retrieval was tested because smaller child chunks can potentially provide more precise matching while the parent provides broader context.

However, the benchmark showed:

* V2 Recall@5: 85%
* V3 Recall@5: 90%
* Final Recall@5: **95%**

Both parent-child versions also failed Q8.

V3 additionally had the highest latency at **2.549s/query**.

Therefore, parent-child retrieval was not selected for the final system.

This does not imply that parent-child chunking is generally inferior. It means that **for this corpus and evaluation set, it did not provide enough retrieval improvement to justify its additional complexity and latency.**

---

# 8. Final Architecture Decision

The final retrieval architecture was selected based on measured performance rather than architectural complexity.

```text
Podcast Audio
      ↓
Transcription
      ↓
300-token chunks + 50 overlap
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

---

# 9. Conclusion

The evaluation followed an iterative process:

```text
Baseline
   ↓
Identify retrieval weaknesses
   ↓
Add CrossEncoder reranking
   ↓
Experiment with parent-child retrieval
   ↓
Evaluate failures and trade-offs
   ↓
Select the highest-performing architecture
```

The final system achieved:

* **95% Recall@5**
* **100% Recall@10**
* **0.707 MRR**

compared with the baseline's:

* **70% Recall@5**
* **100% Recall@10**
* **0.604 MRR**

The main engineering conclusion is that **a simpler flat-chunk architecture combined with candidate generation and CrossEncoder reranking performed best on this evaluation set**.

The experiments also demonstrated that more complex hierarchical retrieval does not automatically produce better retrieval quality. The final architecture was therefore chosen based on reproducible benchmark results and observed failure cases rather than complexity alone.
___________________________________________________________________
## Conversation Checks

A small conversational smoke test was used in addition to retrieval evaluation.

| Scenario                        | Result                                                |
| ------------------------------- | ----------------------------------------------------- |
| Follow-up question              | Correctly resolved using conversation history         |
| Supported podcast question      | Answered with transcript timestamp                    |
| Unsupported question            | Correctly refused                                     |
| Personal conversational context | Correctly remembered the user's stated favorite paper |

The system is designed to remain grounded in the supplied podcast evidence. In
particular, unsupported questions should return a clear refusal rather than
using the model's general knowledge.
### Conversation Failure

The system correctly refused an unsupported question:

> "Can you give me an example of a black hole?"

However, this exposes a product limitation: the assistant is strictly restricted to
the supplied podcast evidence, so it cannot provide a general educational example
even when the requested concept is familiar.

This is safe from a hallucination perspective, but less helpful from a learning
perspective. A future improvement could distinguish between **questions asking
for podcast-grounded facts** and **general educational requests**, while clearly
labeling when an answer comes from outside the supplied episodes.
