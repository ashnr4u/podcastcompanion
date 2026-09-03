import chromadb
from pathlib import Path

# Use absolute path to be safe
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
print("Collections in:", CHROMA_DIR)
print(client.list_collections())