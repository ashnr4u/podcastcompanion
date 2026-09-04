**# Retrieval Evaluation — Fermi Podcast Companion**

**## 1. Product Context**

**### 1.1 What I Built**

The Fermi Podcast Companion is a conversational AI system that allows learners to ask questions about physics podcast episodes and receive grounded, timestamped answers.

****User Persona:**** An undergraduate physics student preparing for oral exams who wants to verify their understanding against primary podcast sources. The student needs both broad coverage (which episodes cover a topic?) and deep engagement (explain this concept, show me the source).

****Core Product Decision:**** I prioritized ****trustworthiness over breadth****. Every answer must be verifiable against the source audio with timestamps, and the system must refuse questions it cannot support.

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

**### 1.2 Why I Evaluated Retrieval**

The system's quality depends entirely on its retrieval stage. If the retrieval system cannot find the right evidence, the LLM cannot produce a grounded answer—no matter how good the generation model is.

Therefore, the evaluation focused on:

> ****Does the retrieval system find the correct evidence, and does it rank it highly enough to be useful?****

---

**## 2. Evaluation Structure**

I built ****two complementary evaluation suites**** because they answer different engineering questions:

| Suite                                        | Question                                                    | Dataset                                          | Evaluator                     |
| -------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------ | ----------------------------- |
| ****A: Controlled Retrieval Benchmark****    | Which retrieval architecture ranks evidence best?           | 20 curated questions with timestamp ground truth | `src/experiments/eval_all.py` |
| ****B: End-to-End Application Evaluation**** | Does the complete application work for realistic questions? | 13 unique cases (3 categories)                   | `run_eval.py`                 |

****Important:**** These suites use different datasets and metrics. Their numerical results should not be directly compared.

---

**## 3. Evaluation Suite A — Controlled Retrieval Benchmark**

**### 3.1 Purpose**

The controlled benchmark evaluates retrieval implementations in isolation from the LLM and answer generation.

It answers:

> ****Which retrieval architecture ranks relevant transcript evidence most effectively?****

**### 3.2 Dataset**

****File:**** `data/evaluate_retrieve.json`

****20 manually curated questions****

Each question has:

* A natural-language query
* Manually annotated ground-truth timestamp range(s)
* The correct source episode

**### 3.3 Relevance Definition**

A retrieved chunk is considered ****relevant**** when its timestamp overlaps a ground-truth interval:

```text
retrieved_start <= correct_end

AND

retrieved_end >= correct_start
```

This is a strict, objective correctness criterion. It tests ****evidence retrieval****, not semantic similarity.

**### 3.4 Metrics**

| Metric              | What It Measures                               | Why It Matters                           |
| ------------------- | ---------------------------------------------- | ---------------------------------------- |
| ****Recall@5****    | % of questions with a relevant chunk in top 5  | The user sees evidence without scrolling |
| ****Recall@10****   | % of questions with a relevant chunk in top 10 | The system has room to recover           |
| ****MRR****         | How highly the first relevant chunk is ranked  | Ranking quality matters for user trust   |
| ****Avg Latency**** | Time for isolated retrieval per query          | Practical system responsiveness          |

**### 3.5 Retrieval Architectures Tested**

I evaluated four architectures as a controlled experiment progression.

| Version    | Implementation          | Key Change                         | Hypothesis                                                 |
| ---------- | ----------------------- | ---------------------------------- | ---------------------------------------------------------- |
| ****V1**** | `retrieve.py`           | Baseline: flat chunks, no reranker | Establish a performance baseline                           |
| ****V2**** | `retrvfinal.py`         | Flat chunks + CrossEncoder         | Reranking improves the ordering of relevant evidence       |
| ****V3**** | `retrieve2.py`          | Parent-child + CrossEncoder        | Smaller chunks improve precision; reranking improves order |
| ****V4**** | `retrieve2_improved.py` | Parent context + CrossEncoder      | Reranker needs broader context to judge relevance          |

**### 3.6 Results**

| Pipeline                         |    Recall@5 |    Recall@10 |           MRR | Avg Latency |
| -------------------------------- | ----------: | -----------: | ------------: | ----------: |
| V1 — Baseline                    |         70% |         100% |         0.604 |      0.097s |
| ****V2 — Flat + CrossEncoder**** | ****95%**** | ****100%**** |         0.707 |      1.993s |
| V3 — Parent-Child                |         85% |          95% |         0.673 |      0.556s |
| V4 — Context-Aware               |         90% |          90% | ****0.735**** |      2.549s |

**### 3.7 Key Findings**

****1. CrossEncoder Reranking Substantially Improved Recall****

```text
Baseline → V2

70% → 95% Recall@5
```

This is a ****25 percentage-point improvement**** — every 4 out of 5 baseline failures were recovered.

****2. The Improvement Had a Measurable Cost****

```text
0.097s → 1.993s
```

The CrossEncoder adds ~1.9 seconds per query. This was an acceptable trade-off for the retrieval quality improvement.

****3. Parent Context Improved MRR But Not Overall Recall****

V4 achieved the highest MRR (0.735) but had:

* Lower Recall@5 (90% vs 95%)
* Higher latency (2.549s vs 1.993s)
* Lower Recall@10 (90% vs 100%)

****Therefore, I selected the simpler flat-chunk + CrossEncoder architecture (V2).****

---

## 4. Evaluation Suite B — End-to-End Application Evaluation

### 4.1 Purpose

The second evaluation measures the complete application, including:
- Query rewriting
- Retrieval + reranking
- LLM answer generation
- Evidence extraction

It answers:

> **Does the complete application retrieve sufficient evidence to answer realistic learner questions?**

### 4.2 Dataset

**File:** `test_case3.json`

**13 unique test cases** across three categories:

| Category | Cases | What It Tests |
|----------|-------|---------------|
| **Multi-source** | 3 | Retrieval across multiple podcast episodes |
| **Deep single-source** | 5 | Deeper retrieval within one episode |
| **Precise / Needle** | 5 | Finding one specific, narrow fact |

### 4.3 Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Source Coverage** | Required sources retrieved / total required | Cross-episode questions need all sources |
| **Evidence Recall@10** | Expected evidence spans matched / total | Correct evidence matters, not just source |
| **Precision@10** | Relevant retrieved chunks / total retrieved | Low precision means noisy context |
| **End-to-End Latency** | Full `app_graph.invoke()` time | Real user experience |
| **Completion Status** | Success / error / failure | System reliability |

### 4.4 Multi-Source Results

| Strategy | Source Coverage | Evidence Recall@10 | Precision@10 | Avg Latency |
|----------|----------------:|-------------------:|-------------:|------------:|
| Episode-Balanced | **0.778** | **0.296** | **0.267** | 43.70s |
| Global Vector | 0.444 | 0.259 | 0.233 | 42.05s |
| Context-Aware Parent-Child | 0.444 | 0.148 | 0.133 | 25.34s |

**Key Finding:** Episode-balanced retrieval improved source coverage from 44.4% to 77.8% — a **33.4 percentage-point improvement**.

However, evidence recall improved only modestly (25.9% → 29.6%). This tells us:

> **Retrieving the correct source is not the same as retrieving the correct evidence.**

The CrossEncoder reranking stage can still favor one episode and push relevant evidence from other required sources outside the final top-10.

### 4.5 Deep Single-Source Results

| Strategy | Source Coverage | Evidence Recall@10 | Precision@10 | Avg Latency |
|----------|----------------:|-------------------:|-------------:|------------:|
| Global Vector | 1.000 | **0.583** | **0.220** | 40.91s |
| Episode-Balanced | 1.000 | 0.517 | 0.140 | 40.73s |
| Context-Aware Parent-Child | 1.000 | 0.367 | 0.180 | **25.10s** |

**Key Finding:** All strategies achieved perfect source coverage (single source required). Global Vector produced the strongest evidence recall and precision.

### 4.6 Precise / Needle Results

| Strategy | Source Coverage | Evidence Recall@10 | Precision@10 | Avg Latency | Completion |
|----------|----------------:|-------------------:|-------------:|------------:|------------|
| Context-Aware Parent-Child | **1.000** | **0.800** | **0.140** | **24.48s** | 5/5 |
| Global Vector | 1.000 | 0.750 | 0.075 | 37.66s | 4/5 |
| Episode-Balanced | 1.000 | 0.600 | 0.060 | 35.85s | 5/5 |

**Key Finding:** Context-Aware Parent-Child performed best on precise retrieval (0.800 evidence recall, 24.48s latency). This suggests that broader parent context is useful when the required evidence is narrow and specific.

### 4.7 Error Analysis

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

### Finding 4 — Parent-Child Is Useful for Some Question Types

Context-Aware Parent-Child achieved 0.800 Evidence Recall on precise/needle questions, but it was not the best overall architecture in the controlled benchmark.

---
## 6. What I Learned From Failure Analysis

### Q8: Unruh Effect and Black Hole Radiation

**Question:** *"What is the Unruh effect and how does it relate to black hole radiation?"*

**First relevant rank:**

| Pipeline                          | First Relevant Rank |
| --------------------------------- | ------------------: |
| V1 — Baseline (Flat)              |                   7 |
| **V2 — Flat + CrossEncoder**      |               **6** |
| V3 — Parent-Child + CrossEncoder  | Not found in top 10 |
| V4 — Context-Aware + CrossEncoder | Not found in top 10 |

**Why this question is difficult:** The explanation is distributed across multiple conceptual steps:

```text
Unruh effect

    ↓

Acceleration produces perceived temperature

    ↓

Acceleration/gravity equivalence

    ↓

Gravitational field near a horizon

    ↓

Connection to Hawking radiation
```

**What I learned:** Parent-child retrieval does not automatically improve retrieval of multi-step conceptual explanations. The final flat-chunk + CrossEncoder architecture recovered the evidence; parent-child did not.

### Multi-Source Questions

Two cases failed to produce meaningful answers:

| Case    | Question                    | Issue                               |
| ------- | --------------------------- | ----------------------------------- |
| Case 01 | Shannon → DNA & Transformer | Retrieved only Shannon evidence     |
| Case 03 | Transformer → Shannon + DNA | Retrieved only Transformer evidence |

**What I learned:** The system retrieves well from one source but struggles when evidence must be combined across three episodes. Episode-balancing improves source coverage but does not guarantee evidence recall.

---

## 7. What I Would Do Next

### 7.1 Diversity-Aware Final Selection

**Current:**
```text
Candidate retrieval → CrossEncoder → Top-10
```

**Proposed:**
```text
Candidate retrieval → CrossEncoder scores → Diversity-aware selection → Top-10
```

Preserve relevance while ensuring required sources remain represented.

### 7.2 Source-Aware Retrieval

When a question requires multiple sources, explicitly allocate candidates across relevant episodes.

### 7.3 Query Decomposition

Complex cross-source questions can be decomposed:

```text
Original question
    ↓
Shannon sub-query
DNA sub-query
Transformer sub-query
    ↓
Separate retrieval per source
    ↓
Combine evidence
```

### 7.4 Query Expansion

Generate multiple retrieval formulations so that concepts expressed differently in the transcript can still be retrieved.

### 7.5 LLM Reliability

Add retry logic and fallback handling for temporary LLM service failures.

---

## 8. Evaluation Limitations

| Limitation | Impact |
|------------|--------|
| **13 unique test cases** | Useful for targeted testing but not enough for broad generalization |
| **Different metric definitions** | Suite A and Suite B metrics are not directly comparable |
| **Source Coverage ≠ Evidence Recall** | A system can reach the correct source but miss the correct evidence |
| **Different latency scopes** | Isolated retrieval (0.1-2.5s) vs full application (24-43s) |

---

## 9. Conclusion

### What I Accomplished

1. **Built a complete end-to-end system** — audio → transcript → chunks → embeddings → retrieval → reranking → grounded answers

2. **Ran controlled experiments** — four retrieval architectures compared on 20 curated questions

3. **Created a rigorous evaluation system** — 13 unique cases, 9 runs, 3 categories, preserved raw results

4. **Achieved measurable improvement** — 70% → 95% Recall@5, 0.604 → 0.707 MRR

5. **Identified the main weakness** — multi-source evidence coverage

6. **Proposed concrete next steps** — diversity-aware selection, query decomposition, source-aware retrieval

### The Core Lesson

> **Retrieving the correct source is not the same as retrieving the correct evidence. A strong RAG system must optimize both evidence relevance and source coverage.**

The system is strong on single-source and precise questions. The next improvement should focus on **diversity-aware, source-aware retrieval** rather than simply changing the chunking strategy.

---

## 10. How the Evaluation Maps to Fermi's Criteria

| Fermi Criteria | How This Evaluation Addresses It |
|----------------|----------------------------------|
| **Define success** | Recall@5, MRR, Source Coverage, Evidence Recall — clear success definitions |
| **Varied evaluation set** | 13 cases across 3 categories (multi-source, deep single, precise) |
| **Repeatable runner** | `python run_eval.py` — one-command execution |
| **Preserve raw outputs** | 9 JSON result files with full evidence and answers |
| **Inspect successes/failures** | Q8 failure analysis, multi-source degradation |
| **One meaningful improvement** | 70% → 95% Recall@5 with CrossEncoder |
| **Re-run and explain** | Baseline-to-final comparison with trade-off analysis |

---

**Evaluation suites:** 2

**Controlled benchmark cases:** 20

**End-to-end unique cases:** 13

**Total evaluation runs:** 9
