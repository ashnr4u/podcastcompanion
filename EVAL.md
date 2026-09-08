# Retrieval Evaluation — Fermi Podcast Companion

## 1. Product Context

### 1.1 What I Built

The Fermi Podcast Companion is a conversational AI system that allows learners to ask questions about physics podcast episodes and receive grounded, timestamped answers.

**User Persona:** An undergraduate physics student preparing for oral exams who wants to verify their understanding against primary podcast sources. The student needs both broad coverage (which episodes cover a topic?) and deep engagement (explain this concept, show me the source).

**Core Product Decision:** I prioritized **trustworthiness over breadth**. Every answer should be verifiable against the source audio with timestamps, and the system should refuse questions it cannot support.

```text
Primary UX flow:

Question
    ↓
Is this supported by the podcasts?
    ↓
    ├── Yes → Answer + Timestamp references
    └── No  → Clear refusal
```

This trade-off was deliberate: a system that occasionally refuses is safer and more trustworthy than one that hallucinates plausible but incorrect answers.

### 1.2 Why I Evaluated Retrieval

The system's quality depends heavily on its retrieval stage. If the retrieval system cannot find the right evidence, the LLM cannot reliably produce a grounded answer from that evidence.

Therefore, the evaluation focused on:

> **Does the retrieval system find the correct evidence, and does it rank it highly enough to be useful?**

---

## 2. Evaluation Structure

I built **two complementary evaluation suites** because they answer different engineering questions:

| Suite                                    | Question                                                    | Dataset                                          | Evaluator                     |
| ---------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------ | ----------------------------- |
| **A: Controlled Retrieval Benchmark**    | Which retrieval architecture ranks evidence best?           | 20 curated questions with timestamp ground truth | `src/experiments/eval_all.py` |
| **B: End-to-End Application Evaluation** | Does the complete application work for realistic questions? | 13 unique cases (3 categories)                   | `run_eval.py`                 |

**Important:** These suites use different datasets and metrics. Their numerical results should not be directly compared.

---

## 3. Evaluation Suite A — Controlled Retrieval Benchmark

### 3.1 Purpose

The controlled benchmark evaluates retrieval implementations **in isolation** from the LLM and answer generation.

It answers:

> **Which retrieval architecture ranks relevant transcript evidence most effectively?**

### 3.2 Dataset

**File:** `data/evaluate_retrieve.json`

**20 manually curated questions**

Each question has:

* A natural-language query
* Manually annotated ground-truth timestamp range(s)
* The correct source episode

**Note:** All 20 questions in Suite A are **single-source** questions. This is intentional — the goal is to isolate and compare chunking and reranking strategies without the complexity of multi-source retrieval.

### 3.3 Relevance Definition

A retrieved chunk is considered **relevant** when its timestamp overlaps a ground-truth interval:

```text
retrieved_start <= correct_end
AND
retrieved_end >= correct_start
```

This provides a reproducible evidence-matching criterion based on timestamps. The underlying timestamp ranges were manually annotated, so the annotation boundaries involve some human judgment. It tests **evidence retrieval**, not semantic similarity.

### 3.4 Metrics

| Metric          | What It Measures                               | Why It Matters                           |
| --------------- | ---------------------------------------------- | ---------------------------------------- |
| **Recall@5**    | % of questions with a relevant chunk in top 5  | The user sees evidence without scrolling |
| **Recall@10**   | % of questions with a relevant chunk in top 10 | The system has room to recover           |
| **MRR**         | How highly the first relevant chunk is ranked  | Ranking quality matters for user trust   |
| **Avg Latency** | Time for isolated retrieval per query          | Practical system responsiveness          |

### 3.5 Retrieval Architectures Tested

I evaluated four architectures as a controlled experiment progression. V1 and V2 use the flat-chunk collection (`fermi_transcripts`). V3 and V4 use the parent-child collection (`fermi_parent_child`).

| Version | Key Change                                           | Hypothesis                                                 |
| ------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| **V1**  | Flat chunks, no reranker                             | Establish a performance baseline                           |
| **V2**  | Flat chunks + CrossEncoder                           | Reranking improves the ordering of relevant evidence       |
| **V3**  | Parent-child + CrossEncoder (child only)             | Smaller chunks improve precision; reranking improves order |
| **V4**  | Parent-child + CrossEncoder (child + parent context) | Reranker needs broader context to judge relevance          |

**This design allows me to isolate three comparisons:**

* **V1 vs V2**: Effect of CrossEncoder on flat chunks
* **V2 vs V3**: Effect of switching from flat to parent-child chunks (both with reranking)
* **V3 vs V4**: Effect of adding parent context to the reranker

### 3.6 Results

| Pipeline                     | Recall@5 | Recall@10 | MRR       | Avg Latency |
| ---------------------------- | -------- | --------- | --------- | ----------- |
| V1 — Baseline                | 70%      | 100%      | 0.604     | 0.097s      |
| **V2 — Flat + CrossEncoder** | **95%**  | **100%**  | 0.707     | 1.993s      |
| V3 — Parent-Child            | 85%      | 95%       | 0.673     | 0.556s      |
| V4 — Context-Aware           | 90%      | 90%       | **0.735** | 2.549s      |

### 3.7 Key Findings

**1. CrossEncoder Reranking Substantially Improved Recall**

```text
Baseline → V2

70% → 95% Recall@5
```

This is a **25 percentage-point improvement** — V2 recovered **5 of the 6 questions missed by the baseline**.

**2. The Improvement Had a Measurable Cost**

```text
0.097s → 1.993s
```

The CrossEncoder adds ~1.9 seconds per query. This was an acceptable trade-off for the retrieval quality improvement.

**3. Parent Context Improved MRR But Not Overall Recall**

V4 achieved the highest MRR (0.735) but had:

* Lower Recall@5 (90% vs 95%)
* Higher latency (2.549s vs 1.993s)
* Lower Recall@10 (90% vs 100%)

**Therefore, I selected the simpler flat-chunk + CrossEncoder architecture (V2) as the primary retrieval architecture for single-source questions.**

---

## 4. Evaluation Suite B — End-to-End Application Evaluation

### 4.1 Purpose

The second evaluation measures the **complete application**, including:

* Query rewriting
* Retrieval + reranking
* LLM answer generation
* Evidence extraction

It answers:

> **Does the complete application retrieve sufficient evidence to answer realistic learner questions?**

### 4.2 Why Different Strategies in Suite B?

**Important:** Suite B tests **different retrieval strategies** than Suite A. This is intentional and serves a specific purpose:

| Aspect                | Suite A                                    | Suite B                                              |
| --------------------- | ------------------------------------------ | ---------------------------------------------------- |
| **Focus**             | Compare chunking + reranking architectures | Compare retrieval strategies for realistic scenarios |
| **Questions**         | 20 curated, single-source                  | 13 realistic, includes multi-source                  |
| **Goal**              | Find the best ranking architecture         | Validate system performance on real use cases        |
| **Strategies Tested** | V1-V4 (chunking variations)                | Global, Episode-Balanced, Context-Aware              |

**Here's the relationship:**

* **Global Vector** uses the same architecture as V2 (flat chunks + CrossEncoder)
* **Context-Aware Parent-Child** uses the same architecture as V4 (parent-context + CrossEncoder)
* **Episode-Balanced** is a **new strategy introduced specifically for Suite B** — it wasn't tested in Suite A because Suite A's single-source questions don't require source balancing

> **"Episode-Balanced isn't an 'improvement' over V4 — it's a different strategy for a different problem (multi-source coverage)."**

### 4.3 Dataset

**File:** `test_case3.json`

**13 unique test cases** across three categories:

| Category               | Cases | What It Tests                              |
| ---------------------- | ----- | ------------------------------------------ |
| **Multi-source**       | 3     | Retrieval across multiple podcast episodes |
| **Deep single-source** | 5     | Deeper retrieval within one episode        |
| **Precise / Needle**   | 5     | Finding one specific, narrow fact          |

### 4.4 Metrics

| Metric                 | Definition                                  | Why It Matters                            |
| ---------------------- | ------------------------------------------- | ----------------------------------------- |
| **Source Coverage**    | Required sources retrieved / total required | Cross-episode questions need all sources  |
| **Evidence Recall@10** | Expected evidence spans matched / total     | Correct evidence matters, not just source |
| **Precision@10**       | Relevant retrieved chunks / total retrieved | Low precision means noisy context         |
| **End-to-End Latency** | Full `app_graph.invoke()` time              | Real user experience                      |
| **Completion Status**  | Success / error / failure                   | System reliability                        |

### 4.5 Retrieval Strategies Tested in Suite B

| Strategy                       | Implementation  | Key Idea                                                                          | Relationship to Suite A                       |
| ------------------------------ | --------------- | --------------------------------------------------------------------------------- | --------------------------------------------- |
| **Global Vector**              | `1retrieval.py` | Retrieve top 30 globally from entire collection, rerank to top 10                 | Same as V2 (flat + CrossEncoder)              |
| **Episode-Balanced**           | `import os.txt` | Retrieve top 6 from **each** episode, combine candidates, rerank to top 10        | **NEW** — designed for multi-source questions |
| **Context-Aware Parent-Child** | `2retrieval.py` | Retrieve child chunks with parent context in metadata, rerank using combined text | Same as V4 (parent-context + CrossEncoder)    |

**Why Episode-Balanced Was Added:**

During development, I noticed that global retrieval often got **dominated by one episode** — if one episode had more relevant-sounding chunks, it would crowd out evidence from other required sources. Episode-Balanced addresses this by:

1. **Forcing diversity** — retrieving from every episode
2. **Improving source coverage** — no single episode can completely dominate the candidate pool
3. **Preserving relevance** — reranking still selects the best chunks overall

### 4.6 Multi-Source Results

| Strategy                   | Source Coverage | Evidence Recall@10 | Precision@10 | Avg Latency |
| -------------------------- | --------------- | ------------------ | ------------ | ----------- |
| Episode-Balanced           | **0.778**       | **0.296**          | **0.267**    | 43.70s      |
| Global Vector              | 0.444           | 0.259              | 0.233        | 42.05s      |
| Context-Aware Parent-Child | 0.444           | 0.148              | 0.133        | 25.34s      |

**Key Finding:** Episode-balanced retrieval improved source coverage from 44.4% to 77.8% — a **33.4 percentage-point improvement**.

However, evidence recall improved only modestly (25.9% → 29.6%). This tells us:

> **Retrieving the correct source is not the same as retrieving the correct evidence.**

The CrossEncoder reranking stage can still favor one episode and push relevant evidence from other required sources outside the final top-10.

Because the multi-source subset contains only **3 cases**, this result should be treated as a useful evaluation signal rather than broad evidence that Episode-Balanced is universally superior.

### 4.7 Deep Single-Source Results

| Strategy                   | Source Coverage | Evidence Recall@10 | Precision@10 | Avg Latency |
| -------------------------- | --------------- | ------------------ | ------------ | ----------- |
| Global Vector              | 1.000           | **0.583**          | **0.220**    | 40.91s      |
| Episode-Balanced           | 1.000           | 0.517              | 0.140        | 40.73s      |
| Context-Aware Parent-Child | 1.000           | 0.367              | 0.180        | **25.10s**  |

**Key Finding:** All strategies achieved perfect source coverage (single source required). Global Vector produced the strongest evidence recall and precision — consistent with V2's stronger performance in Suite A for single-source retrieval.

### 4.8 Precise / Needle Results

| Strategy                   | Source Coverage | Evidence Recall@10 | Precision@10 | Avg Latency | Completion |
| -------------------------- | --------------- | ------------------ | ------------ | ----------- | ---------- |
| Context-Aware Parent-Child | **1.000**       | **0.800**          | **0.140**    | **24.48s**  | 5/5        |
| Global Vector              | 1.000           | 0.750              | 0.075        | 37.66s      | 4/5        |
| Episode-Balanced           | 1.000           | 0.600              | 0.060        | 35.85s      | 5/5        |

**Key Finding:** Context-Aware Parent-Child performed best on precise retrieval (0.800 evidence recall, 24.48s latency). This suggests that broader parent context is useful when the required evidence is narrow and specific — although this result is based on only five cases.

### 4.9 Error Analysis

One execution encountered an `LLMServiceError`:

```text
Case 14alt (Black hole entropy ratio)

Strategy: Global Vector

Error: LLMServiceError — language model service failure
```

**Important:** This was a **generation service failure**, not a retrieval failure.

```text
Retrieval failure ≠ Generation failure ≠ Service failure
```

Future improvements should include retry logic and fallback handling for temporary LLM service failures.

---

## 5. Key Findings Across Both Suites

### Finding 1 — CrossEncoder Improves Controlled Retrieval

The controlled benchmark showed that flat chunks + CrossEncoder achieved:

```text
Recall@5  = 95%
Recall@10 = 100%
MRR       = 0.707
```

### Finding 2 — Multi-Source Retrieval Is the Main Weakness

The end-to-end evaluation showed that cross-episode questions are significantly more difficult than single-source questions. Global retrieval can become dominated by one episode.

### Finding 3 — Episode Balancing Improves Source Coverage

Episode-balanced candidate retrieval increased multi-source Source Coverage:

```text
44.4% → 77.8%
```

But Evidence Recall increased less:

```text
25.9% → 29.6%
```

### Finding 4 — Different Strategies Excel at Different Tasks

| Strategy                            | Best Use Case                       | Evidence              |
| ----------------------------------- | ----------------------------------- | --------------------- |
| **V2 / Global Vector**              | Single-source, conceptual questions | 95% Recall@5          |
| **V4 / Context-Aware Parent-Child** | Precise/needle questions            | 0.800 Evidence Recall |
| **Episode-Balanced**                | Multi-source questions              | 77.8% Source Coverage |

> **"There's no single 'best' strategy — each excels at a different query type."**

---
## 6. Failure Analysis and Key Findings

The evaluation exposed two important retrieval limitations. First, the Q8 Unruh-effect question showed that parent-child retrieval did not consistently improve multi-step conceptual retrieval; the flat-chunk + CrossEncoder configuration performed better on this case. Second, multi-source questions revealed that global retrieval could be dominated by a single episode, causing evidence from other required episodes to be excluded.

Episode-Balanced retrieval addressed the second problem by increasing **Source Coverage from 44.4% to 77.8%** on the multi-source subset. However, **Evidence Recall@10 increased only from 25.9% to 29.6%**, showing that retrieving the correct source does not necessarily retrieve the required evidence.

## 7. Final Retrieval Implementation

Based on the combined evaluation, the final system uses **flat chunks + CrossEncoder reranking with Episode-Balanced retrieval**.

| Component          | Final Choice                         | Rationale                                                 |
| ------------------ | ------------------------------------ | --------------------------------------------------------- |
| Chunking           | Flat chunks (300 tokens, 50 overlap) | 95% Recall@5 in Suite A                                   |
| Reranking          | CrossEncoder                         | Improved Recall@5 from 70% to 95%                         |
| Retrieval strategy | Episode-Balanced                     | Improved multi-source Source Coverage from 44.4% to 77.8% |

The final pipeline is:

```text
User Query
    ↓
Episode-Balanced Retrieval
    ↓
Top 6 candidates per episode
    ↓
CrossEncoder Reranking
    ↓
Top 10 chunks
    ↓
LLM Answer Generation
    ↓
Grounded Answer + Timestamps
```

Episode-Balanced was selected because the primary weakness identified during application testing was multi-source retrieval. This choice involves a trade-off: Global Vector remained stronger for deep single-source retrieval, while Episode-Balanced provided substantially better source coverage for multi-source questions.

## 8. Next Steps

The evaluation suggests three concrete improvements:

1. **Diversity-aware final selection** — preserve relevance while ensuring required sources remain represented in the final context.
2. **Query decomposition** — split multi-source questions into source-specific retrieval queries before combining the evidence.
3. **Larger multi-source benchmark** — validate the observed improvement beyond the current three multi-source cases.

## 9. Evaluation Limitations

| Limitation                        | Impact                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------- |
| 13 unique Suite B cases           | Insufficient for broad generalization                                           |
| Only 3 multi-source cases         | Episode-Balanced's advantage requires further validation                        |
| Manual timestamp annotations      | Ground-truth boundaries involve human judgment                                  |
| Source Coverage ≠ Evidence Recall | Correct source retrieval does not guarantee correct evidence retrieval          |
| Different latency scopes          | Suite A measures isolated retrieval; Suite B measures application-level latency |
| Suite A is single-source          | Source-balancing strategies cannot be evaluated in Suite A                      |

## 10. Conclusion

The evaluation produced three main findings:

1. **CrossEncoder reranking substantially improved controlled retrieval**, increasing Recall@5 from 70% to 95%.
2. **Multi-source retrieval was the main weakness** of the application. Episode-Balanced retrieval increased Source Coverage from 44.4% to 77.8%, although Evidence Recall@10 improved only modestly from 25.9% to 29.6%.
3. **Retrieval strategy performance depends on query type**: Global Vector performed best for deep single-source retrieval, Context-Aware Parent-Child performed best on the precise/needle subset, and Episode-Balanced performed best for multi-source source coverage.

The final system therefore uses **flat chunks + CrossEncoder reranking with Episode-Balanced retrieval**, while recognizing that further work is needed to improve evidence-level recall for complex multi-source questions.

### Evaluation Summary

* **Evaluation suites:** 2
* **Suite A cases:** 20
* **Suite B cases:** 13
* **Suite B categories:** 3
* **Evaluation runs:** 9
* **Best controlled Recall@5:** 95%
* **Multi-source Source Coverage:** 44.4% → 77.8%
