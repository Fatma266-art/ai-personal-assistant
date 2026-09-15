from pathlib import Path
from tqdm import tqdm
import json
import re


# =========================
# Paths
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "Data" / "Processed Data"
)

CHUNKS_DIR = (
    PROCESSED_DATA_DIR / "chunks"
)

CHUNKS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Chunk Settings
# =========================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# =========================
# Helpers
# =========================

def get_page_number(text: str) -> int | None:
    """
    Get the page number from the beginning of a text block.
    """

    match = re.search(
        r"\[PAGE\s+(\d+)\]",
        text
    )

    if match:
        return int(match.group(1))

    return None


def clean_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove PDF figure tokens
    text = text.replace("<EOS>", "")
    text = text.replace("<pad>", "")

    # Fix words broken across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove Attention Visualizations section
    if "Attention Visualizations" in text:
        text = text.split("Attention Visualizations")[0]

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================
# Chunking
# =========================

def get_overlap_text(text: str, overlap: int) -> str:
    """
    Return the tail of `text` (up to `overlap` chars),
    trimmed to the nearest word boundary so we don't
    cut a word in half.
    """
    if len(text) <= overlap:
        return text

    tail = text[-overlap:]
    space_idx = tail.find(" ")

    if space_idx != -1:
        tail = tail[space_idx + 1:]

    return tail

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:

    """
    Split text into paragraph-aware chunks
    while preserving page information and
    applying a real character-based overlap.
    """

    text = clean_text(text)

    # --------------------------------
    # Split document into pages
    # --------------------------------
    page_blocks = re.split(
        r"(?=\[PAGE\s+\d+\])",
        text
    )

    chunks = []
    current_chunk = ""
    current_page = None

    for page_block in page_blocks:

        page_block = page_block.strip()

        if not page_block:
            continue

        # Get page number
        page_number = get_page_number(page_block)

        if page_number is not None:
            current_page = page_number

        # Remove PAGE marker
        page_text = re.sub(
            r"^\[PAGE\s+\d+\]\s*",
            "",
            page_block
        ).strip()

        if not page_text:
            continue

        # --------------------------------
        # Split page into paragraphs
        # --------------------------------
        paragraphs = re.split(
            r"\n\s*\n",
            page_text
        )

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # --------------------------------
            # Handle very large paragraph
            # --------------------------------
            if len(paragraph) > chunk_size:

                # Flush current chunk first
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": current_page
                    })

                # Split large paragraph into
                # fixed-size pieces with overlap
                start = 0

                while start < len(paragraph):

                    end = min(
                        start + chunk_size,
                        len(paragraph)
                    )

                    chunk_text = paragraph[start:end].strip()

                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "page": current_page
                        })

                    # Move forward while keeping overlap
                    next_start = end - overlap

                    if next_start <= start:
                        break

                    start = next_start

                # Reset current chunk
                current_chunk = ""

                continue

            # --------------------------------
            # Normal paragraph
            # --------------------------------
            candidate = (
                current_chunk + "\n\n" + paragraph
                if current_chunk
                else paragraph
            )

            # --------------------------------
            # Paragraph fits
            # --------------------------------
            if len(candidate) <= chunk_size:

                current_chunk = candidate

            # --------------------------------
            # Paragraph does NOT fit
            # --------------------------------
            else:

                # Save previous chunk
                if current_chunk:

                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": current_page
                    })

                # Get overlap from previous chunk
                overlap_text = get_overlap_text(
                    current_chunk,
                    overlap
                )

                # Start new chunk with overlap
                if overlap_text:
                    current_chunk = (
                        overlap_text
                        + "\n\n"
                        + paragraph
                    )
                else:
                    current_chunk = paragraph

                # Safety check
                if len(current_chunk) > chunk_size:

                    current_chunk = paragraph

    # --------------------------------
    # Save final chunk
    # --------------------------------
    if current_chunk:

        chunks.append({
            "text": current_chunk.strip(),
            "page": current_page
        })

    return chunks


# =========================
# Process All Text Files
# =========================

def process_all_texts():

    txt_files = list(
        PROCESSED_DATA_DIR.glob("*.txt")
    )

    if not txt_files:

        print(
            "No .txt files found in "
            "Processed Data."
        )

        return

    print(
        f"Found {len(txt_files)} text file(s)\n"
    )

    all_chunks = []

    for txt_path in tqdm(
        txt_files,
        desc="Chunking"
    ):

        print(
            f"\n→ Processing: {txt_path.name}"
        )

        text = txt_path.read_text(
            encoding="utf-8"
        )

        chunks = split_text(text)

        print(
            f"   Created {len(chunks)} chunks"
        )

        # --------------------------------
        # Add metadata
        # --------------------------------

        for i, chunk in enumerate(chunks):

            chunk_data = {

                "source": txt_path.stem,

                "chunk_id": i,

                "page": chunk["page"],

                "text": chunk["text"]

            }

            all_chunks.append(
                chunk_data
            )

    # =========================
    # Save
    # =========================

    output_path = (
        CHUNKS_DIR / "all_chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\n✅ Total chunks created: "
        f"{len(all_chunks)}"
    )

    print(
        f"✅ Saved to: {output_path}"
    )


if __name__ == "__main__":
    process_all_texts()