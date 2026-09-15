from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch 

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CHUNKS_PATH = PROJECT_ROOT / "Data" / "Processed Data" / "chunks" / "all_chunks.json"
CHROMA_PATH = PROJECT_ROOT / "Data" / "Processed Data" / "chroma_db"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# 3. Settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
COLLECTION_NAME = "knowledge_base"
BATCH_SIZE = 64 


def load_chunks(chunks_path: Path) -> List[Dict]:
    """Load chunks from JSON file."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}. Run chunk_text.py first.")

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    logging.info(f"Loaded {len(chunks)} chunks")
    return chunks


def get_device() -> str:
    """Check if GPU is available."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_vectorstore():
    """Create embeddings and store them in ChromaDB."""

    # 1. Load chunks
    chunks = load_chunks(CHUNKS_PATH)

    # 2. Load embedding model
    device = get_device()
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    # 3. Initialize ChromaDB
    logging.info(f"Initializing ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete collection if it already exists
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(name=COLLECTION_NAME)
        logging.info("Old collection deleted.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Create embeddings and add to Chroma in batches
    logging.info("Creating embeddings and storing in ChromaDB...")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[i : i + BATCH_SIZE]

        texts = [item["text"] for item in batch]
        ids = [f"{item['source']}_{item['chunk_id']}" for item in batch]
        metadatas = [
         {
            "source": item["source"],
            "chunk_id": str(item["chunk_id"]),
            "page": item.get("page", -1)
         }
         for item in batch
        ]

        embeddings = model.encode(
            texts, 
            show_progress_bar=False, 
            batch_size=BATCH_SIZE
        ).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    logging.info("✅ Done!")
    logging.info(f"✅ Total documents in collection: {collection.count()}")
    logging.info(f"✅ ChromaDB saved at: {CHROMA_PATH}")


if __name__ == "__main__":
    try:
        create_vectorstore()
    except Exception as e:
        logging.exception(f"An error occurred: {e}")