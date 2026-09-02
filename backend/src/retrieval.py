import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


# ============================================================
# Configuration
# ============================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PER_EPISODE_K = 6
RETRIEVAL_K = 30
TOP_K = 10


# ============================================================
# Runtime Resources
# ============================================================

embedder = None
reranker = None
client = None
collection = None


# ============================================================
# Initialization
# ============================================================

def initialize():

    global embedder
    global reranker
    global client
    global collection

    # Prevent loading everything twice
    if (
        embedder is not None
        and reranker is not None
        and collection is not None
    ):
        return

    print("\n" + "=" * 60)
    print("INITIALIZING RETRIEVAL SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    if embedder is None:

        print("Loading embedding model...")

        embedder = SentenceTransformer(
            EMBED_MODEL_NAME
        )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    if reranker is None:

        print("Loading reranker...")

        reranker = CrossEncoder(
            RERANK_MODEL_NAME
        )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    if collection is None:

        print("Connecting to ChromaDB...")

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        collection = client.get_collection(
            name="fermi_transcripts"
        )

        print(
            f"Connected to ChromaDB: {CHROMA_DIR}"
        )

        print(
            f"Total chunks: {collection.count()}"
        )

    print("Retrieval system ready.")
    print("=" * 60 + "\n")


# ============================================================
# Discover Episodes
# ============================================================

def get_episodes():

    initialize()

    result = collection.get(
        include=["metadatas"]
    )

    episodes = set()

    for metadata in result["metadatas"]:

        if metadata and metadata.get("video_id"):

            episodes.add(
                metadata["video_id"]
            )

    return sorted(episodes)


# ============================================================
# Step 1: Episode-Balanced Vector Retrieval
# ============================================================

def vector_search(
    query,
    per_episode_k=PER_EPISODE_K
):

    initialize()

    query_embedding = embedder.encode(
        query
    ).tolist()

    episodes = get_episodes()

    chunks = []

    for episode in episodes:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=per_episode_k,
            where={
                "video_id": episode
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(documents)):

            chunks.append({
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            })

    print(
        f"Retrieved {len(chunks)} candidates "
        f"across {len(episodes)} episodes"
    )

    return chunks


# ============================================================
# Step 2: Cross-Encoder Reranking
# ============================================================

def rerank(
    query,
    chunks,
    top_k=TOP_K
):

    initialize()

    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    for chunk, score in zip(
        chunks,
        scores
    ):

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

    initialize()

    candidates = vector_search(
        query,
        PER_EPISODE_K
    )

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

    initialize()

    while True:

        query = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if query.lower().strip() == "exit":
            break

        chunks = retrieve(query)

        print(
            "\n========== FINAL RETRIEVED RESULTS ==========\n"
        )

        for i, chunk in enumerate(
            chunks,
            1
        ):

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

