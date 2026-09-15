"""
Single-File Ingestion
----------------------
Takes one uploaded PDF, extracts text, chunks it, embeds it,
and adds it to the existing ChromaDB collection —
without touching what's already indexed.
"""

from pathlib import Path
import logging

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch

from src.ingestion.extract_text import extract_text_from_pdf
from src.ingestion.chunk_text import split_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CHROMA_PATH = PROJECT_ROOT / "Data" / "Processed Data" / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "knowledge_base"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ingest_file(pdf_path: Path, model: SentenceTransformer = None) -> int:
    """
    Extract, chunk, embed, and index a single supported document into the
    existing ChromaDB collection. Returns the number of chunks added.
    Safe to call multiple times on the same file — it overwrites
    that file's own chunks, not the whole collection.
    """
    logger.info(f"Ingesting: {pdf_path.name}")

    # 1. Extract
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        logger.warning(f"No extractable text found in {pdf_path.name}")
        return 0

    # 2. Chunk
    chunks = split_text(text)
    if not chunks:
        return 0

    source_name = pdf_path.stem

    # 3. Use the passed-in model if we have one, otherwise load a new one
    if model is None:
        device = get_device()
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Remove any previous chunks from this same file
    existing_ids = collection.get(
        where={"source": source_name}
    ).get("ids", [])

    if existing_ids:
        collection.delete(ids=existing_ids)
        logger.info(f"Removed {len(existing_ids)} old chunks for {source_name}")

    # 5. Embed and add new chunks
    texts = [c["text"] for c in chunks]
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": source_name,
            "chunk_id": str(i),
            "page": c.get("page", -1)
        }
        for i, c in enumerate(chunks)
    ]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"✅ Added {len(chunks)} chunks from {pdf_path.name}")
    return len(chunks)
