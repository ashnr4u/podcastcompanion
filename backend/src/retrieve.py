import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "data" / "chroma"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10

# -----------------------------
# Load model
# -----------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="fermi_transcripts"
)

print(f"Connected to ChromaDB: {CHROMA_DIR}")
print(f"Total chunks: {collection.count()}")


# -----------------------------
# Retrieval
# -----------------------------

def retrieve_chunks(query, top_k=TOP_K):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []

    for i in range(len(results["documents"][0])):

        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return chunks


# -----------------------------
# Interactive query
# -----------------------------

if __name__ == "__main__":

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        chunks = retrieve_chunks(query)

        print("\n========== TOP RESULTS ==========\n")

        for i, chunk in enumerate(chunks, 1):

            metadata = chunk["metadata"]

            print(f"RESULT {i}")
            print(f"Video: {metadata['title']}")
            print(
                f"Timestamp: "
                f"{metadata['start']:.2f}s → "
                f"{metadata['end']:.2f}s"
            )
            print(f"Distance: {chunk['distance']:.4f}")
            print(f"Text: {chunk['text']}")
            print("-" * 70)