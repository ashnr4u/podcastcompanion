# Retrieval Evaluation Report

## Objective

Evaluate whether adding a cross-encoder reranker improves the baseline
dense vector retrieval system.

## Setup

- Evaluation dataset: 20 question–ground-truth chunk pairs
- **V1:** Dense Vector Search using ChromaDB
- **V2:** Dense Vector Search + Cross-Encoder Reranker
- Metrics: Recall@5, Recall@10, MRR, Average Latency

## Results

| Metric | V1 - Vector | V2 - Vector + Reranker |
|---|---:|---:|
| Recall@5 | 70% | **95%** |
| Recall@10 | 100% | 100% |
| MRR | 0.604 | **0.707** |
| Avg Latency | 0.121s | 2.555s |

## Analysis

V2 substantially improves the quality of the top-ranked results.

Recall@5 increased from **70% to 95%**, meaning the correct chunk was
found within the top 5 for 19/20 questions, compared with 14/20 for V1.

MRR also improved from **0.604 to 0.707**, showing that relevant chunks
were generally ranked higher.

Recall@10 remained at 100% for both systems, so the main benefit of
reranking is **better ordering of retrieved candidates**, rather than
finding information that vector search could not retrieve.

The trade-off is latency: V2 is significantly slower because of the
additional cross-encoder reranking step.

## Conclusion

The reranker provides a clear retrieval-quality improvement:

**Recall@5: 70% → 95%**  
**MRR: 0.604 → 0.707**

The next experiment is to evaluate **V3: Hybrid Retrieval + Reranker**
using the same evaluation dataset.