from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm
import logging

# =========================
# Setup
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw Data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "Processed Data"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# PDF Extraction
# =========================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from PDF while preserving page boundaries.
    Each page starts with a [PAGE X] marker.
    """
    try:
        with fitz.open(pdf_path) as doc:
            pages = []

            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text")

                if not text.strip():
                    logging.warning(f"Page {page_number} in {pdf_path.name} is empty (might be scanned image)")
                    continue

                page_text = text.strip()
                pages.append(
                    f"[PAGE {page_number}]\n"
                    f"{page_text}"
                )

            return "\n\n".join(pages)

    except Exception as e:
        logging.error(f"Failed to extract from {pdf_path.name}: {e}")
        return ""

# =========================
# Process PDFs
# =========================

def process_all_pdfs():
    """Process all PDF files in the Raw Data directory."""
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        logging.warning("No PDF files found in Raw Data directory.")
        return

    logging.info(f"Found {len(pdf_files)} PDF file(s)")

    for pdf_path in tqdm(pdf_files, desc="Extracting text"):
        logging.info(f"Processing: {pdf_path.name}")

        text = extract_text_from_pdf(pdf_path)

        if text:
            output_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}.txt"

            output_path.write_text(
                text,
                encoding="utf-8"
            )

            logging.info(f"Saved: {output_path.name} | Characters: {len(text)}")

if __name__ == "__main__":
    process_all_pdfs()