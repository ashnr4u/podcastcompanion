import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"


# ============================================================
# Configuration
# ============================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieve a larger candidate pool first
RETRIEVAL_K = 30

# Only the best chunks are passed to the LLM
TOP_K = 10


# ============================================================
# Load Models
# ============================================================

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL_NAME)


# ============================================================
# Connect to ChromaDB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="fermi_transcripts"
)

print(f"Connected to ChromaDB: {CHROMA_DIR}")
print(f"Total chunks: {collection.count()}")


# ============================================================
# Step 1: Vector Retrieval
# ============================================================

def vector_search(query, top_k=RETRIEVAL_K):

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    chunks = []

    for i in range(len(results["documents"][0])):

        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return chunks


# ============================================================
# Step 2: Cross-Encoder Reranking
# ============================================================

def rerank(query, chunks, top_k=TOP_K):

    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    chunks = sorted(
        chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return chunks[:top_k]


# ============================================================
# Complete Retrieval Pipeline
# ============================================================

def retrieve(query):

    # Stage 1: Fast semantic retrieval
    candidates = vector_search(
        query,
        RETRIEVAL_K
    )

    # Stage 2: More precise reranking
    final_chunks = rerank(
        query,
        candidates,
        TOP_K
    )

    return final_chunks


# ============================================================
# Interactive Testing
# ============================================================

if __name__ == "__main__":

    while True:

        query = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if query.lower().strip() == "exit":
            break

        chunks = retrieve(query)

        print("\n========== FINAL RETRIEVED RESULTS ==========\n")

        for i, chunk in enumerate(chunks, 1):

            metadata = chunk["metadata"]

            print(f"RESULT {i}")

            print(
                f"Video: {metadata['title']}"
            )

            print(
                f"Timestamp: "
                f"{metadata['start']:.2f}s → "
                f"{metadata['end']:.2f}s"
            )

            print(
                f"Vector Distance: "
                f"{chunk['distance']:.4f}"
            )

            print(
                f"Rerank Score: "
                f"{chunk['rerank_score']:.4f}"
            )

            print(
                f"Text: {chunk['text']}"
            )

            print("-" * 70)