"""
Retrieval Module - RAG Pipeline
--------------------------------
This module handles semantic search over the vector store.
It loads the embedding model and ChromaDB collection once,
then retrieves the most relevant chunks for any given query.
"""

from pathlib import Path
import logging
import os
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch

# Load environment variables
load_dotenv()

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 2. Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CHROMA_PATH = PROJECT_ROOT / "Data" / "Processed Data" / "chroma_db"


# 3. Settings (can be overridden from .env)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
TOP_K = int(os.getenv("TOP_K", "5"))


def get_device() -> str:
    """Return the best available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_retriever() -> Tuple[SentenceTransformer, chromadb.Collection]:
    """
    Load embedding model and Chroma collection once (cached).
    Raises a clear error if the collection does not exist.
    """
    device = get_device()
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on {device}...")

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

    logger.info(f"Connecting to ChromaDB at: {CHROMA_PATH}")

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB path not found: {CHROMA_PATH}\n"
            "Please run create_vectorstore.py first."
        )

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise

    # Check if collection exists
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing_collections:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found.\n"
            f"Available collections: {existing_collections}\n"
            "Please run create_vectorstore.py first."
        )

    collection = client.get_collection(name=COLLECTION_NAME)
    logger.info(f"Collection loaded successfully. Total documents: {collection.count()}")

    return model, collection


def retrieve(
    query: str,
    model: Optional[SentenceTransformer] = None,
    collection: Optional[chromadb.Collection] = None,
    top_k: int = TOP_K
) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: User question or search text.
        model: Optional pre-loaded SentenceTransformer model.
        collection: Optional pre-loaded Chroma collection.
        top_k: Number of results to return.

    Returns:
        List of dictionaries containing rank, text, source, chunk_id, distance, and score.
    """
    if not query or not query.strip():
        logger.warning("Empty query received.")
        return []

    # Load if not provided (uses cache)
    if model is None or collection is None:
        model, collection = load_retriever()

    try:
        # Embed the query
        query_embedding = model.encode(query, show_progress_bar=False).tolist()

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise

    # Format results safely

    retrieved = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else 999.0

        item = {
            "rank": i + 1,
            "text": documents[i] if i < len(documents) else "",
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", "unknown"),
            "page": meta.get("page", "unknown"),
            "distance": distance,
            "score": 1 / (1 + distance)
        }

        retrieved.append(item)

    return retrieved


def print_results(query: str, results: List[Dict]) -> None:
    """Pretty print retrieval results."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for r in results:
        print(
            f"\n--- Rank {r['rank']} | "
            f"Score: {r['score']:.4f} | "
            f"Source: {r['source']} | "
            f"Page: {r['page']} ---"
        )
        print(r["text"][:500])
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        # Load once (cached)
        model, collection = load_retriever()

        test_queries = [
            "What is the attention mechanism?",
            "How does GPT-3 work?",
            "What is Retrieval Augmented Generation?",
        ]

        for q in test_queries:
            results = retrieve(q, model=model, collection=collection, top_k=TOP_K)
            print_results(q, results)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")