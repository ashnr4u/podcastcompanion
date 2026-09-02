import json
from pathlib import Path

import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRANSCRIPT_DIR = BASE_DIR / "data" / "raw_transcripts"
CHROMA_DIR = BASE_DIR / "data" / "chroma"


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

ENCODER = tiktoken.get_encoding("cl100k_base")


# -----------------------------
# Load transcripts
# -----------------------------

def load_transcripts():
    """
    Load all transcript JSON files from data/raw_transcripts/.
    """

    transcripts = []

    if not TRANSCRIPT_DIR.exists():
        print(f"Transcript directory not found: {TRANSCRIPT_DIR}")
        return transcripts

    for path in sorted(TRANSCRIPT_DIR.glob("*.json")):

        print(f"Loading: {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        transcripts.append(transcript)

    return transcripts


# -----------------------------
# Create chunks
# -----------------------------

def create_chunks(transcript):
    """
    Convert transcript segments into
    300-token chunks with 50-token overlap.

    Timestamp information is preserved
    for each chunk.
    """

    segments = transcript["segments"]

    all_tokens = []

    for segment in segments:

        tokens = ENCODER.encode(
            segment["text"]
        )

        for token in tokens:

            all_tokens.append({
                "token": token,
                "start": segment["start"],
                "end": (
                    segment["start"]
                    + segment.get("duration", 0)
                )
            })

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start_idx in range(
        0,
        len(all_tokens),
        step
    ):

        chunk_tokens = all_tokens[
            start_idx:start_idx + CHUNK_SIZE
        ]

        if not chunk_tokens:
            break

        token_ids = [
            item["token"]
            for item in chunk_tokens
        ]

        text = ENCODER.decode(token_ids).strip()

        if not text:
            continue

        start_time = chunk_tokens[0]["start"]
        end_time = chunk_tokens[-1]["end"]

        chunks.append({
            "text": text,
            "start": start_time,
            "end": end_time
        })

        # Stop after final chunk
        if start_idx + CHUNK_SIZE >= len(all_tokens):
            break

    return chunks


# -----------------------------
# Ingestion
# -----------------------------

def ingest_transcripts():
    """
    Load transcripts, create chunks,
    generate embeddings and store them
    in persistent ChromaDB.

    This function is called by main.py.
    """

    print("\nStarting ingestion...\n")

    transcripts = load_transcripts()

    if not transcripts:
        print("No transcripts found.")
        return

    # -----------------------------
    # Load embedding model
    # -----------------------------

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------
    # Create ChromaDB
    # -----------------------------

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="fermi_transcripts"
    )

    total_chunks = 0

    # -----------------------------
    # Process transcripts
    # -----------------------------

    for transcript in transcripts:

        filename = transcript["file"]

        video_id = Path(filename).stem

        print(f"\nProcessing: {filename}")

        chunks = create_chunks(transcript)

        print(
            f"Created {len(chunks)} chunks "
            f"(300 tokens, 50 token overlap)"
        )

        if not chunks:
            print("No chunks generated. Skipping.")
            continue

        texts = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):

            texts.append(
                chunk["text"]
            )

            ids.append(
                f"{video_id}_{i}"
            )

            metadatas.append({
                "video_id": video_id,
                "title": filename,
                "start": chunk["start"],
                "end": chunk["end"]
            })

        # -----------------------------
        # Generate embeddings
        # -----------------------------

        print("Generating embeddings...")

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        ).tolist()

        # -----------------------------
        # Store in ChromaDB
        # -----------------------------

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(chunks)

        print(
            f"Stored {len(chunks)} chunks in ChromaDB."
        )

    # -----------------------------
    # Summary
    # -----------------------------

    print("\n" + "-" * 40)
    print("Ingestion complete")
    print("-" * 40)

    print(f"Processed chunks: {total_chunks}")
    print(f"ChromaDB chunks: {collection.count()}")
    print(f"Chroma DB: {CHROMA_DIR}")


# -----------------------------
# Standalone execution
# -----------------------------

if __name__ == "__main__":
    ingest_transcripts()