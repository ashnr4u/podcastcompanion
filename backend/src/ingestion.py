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


#returns a list of all transcript JSON objects found in data/raw_transcripts/.
def load_transcripts():

    transcripts = []

    for path in TRANSCRIPT_DIR.glob("*.json"):
        
        print(f"Loading: {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        transcripts.append(transcript)
    return transcripts


# -----------------------------
# Create 300-token chunks
# with 50-token overlap
# -----------------------------

def create_chunks(transcript):

    segments = transcript["segments"]

    # Combine transcript while keeping segment timestamps
    all_tokens = []

    for segment in segments:

        tokens = ENCODER.encode(segment["text"])

        for token in tokens:

            all_tokens.append({
                "token": token,
                "start": segment["start"],
                "end": segment["start"] + segment.get("duration", 0)
            })

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start_idx in range(0, len(all_tokens), step):

        chunk_tokens = all_tokens[
            start_idx:start_idx + CHUNK_SIZE
        ]

        if not chunk_tokens:
            break

        # Convert tokens back to text
        token_ids = [
            item["token"]
            for item in chunk_tokens
        ]

        text = ENCODER.decode(token_ids)

        # Timestamp range
        start_time = chunk_tokens[0]["start"]
        end_time = chunk_tokens[-1]["end"]

        chunks.append({
            "text": text.strip(),
            "start": start_time,
            "end": end_time
        })

        # Stop once we reach the end
        if start_idx + CHUNK_SIZE >= len(all_tokens):
            break

    return chunks


# -----------------------------
# Main ingestion
# -----------------------------

def main():

    print("Starting ingestion...\n")

    transcripts = load_transcripts()

    if not transcripts:
        print("No transcripts found.")
        return

    # Load embedding model once
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # Create persistent Chroma database
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="fermi_transcripts"
    )

    total_chunks = 0

    # Process each transcript
    for transcript in transcripts:
        filename = transcript["file"]
        video_id = Path(filename).stem
        print(f"\nProcessing: {filename}")
        chunks = create_chunks(transcript)
        print(
            f"Created {len(chunks)} chunks "
            f"(300 tokens, 50 token overlap)"
        )

        texts = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):

            texts.append(chunk["text"])

            ids.append(
                f"{video_id}_{i}"
            )

            metadatas.append({
                "video_id": video_id,
                "title": filename,
                "start": chunk["start"],
                "end": chunk["end"]
            })

        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        ).tolist()

        # Store in Chroma
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(chunks)

    print("\n-----------------------------")
    print("Ingestion complete")
    print("-----------------------------")
    print(f"Total chunks: {total_chunks}")
    print(f"Chroma DB: {CHROMA_DIR}")


if __name__ == "__main__":
    main()