from pathlib import Path
import json
import logging
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.extract_text import extract_text
from src.llm.reason import get_openai_client, LLM_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CV_RAW_DIR = PROJECT_ROOT / "Data" / "Raw Data" / "CVs"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_document_text(file_path: Path) -> str:
    """Backward-compatible entry point for extracting a CV or profile document."""
    return extract_text(file_path)


EXTRACTION_PROMPT = """Extract the following fields from this CV.
Return ONLY valid JSON, no explanation, no markdown fences.

Fields:
- name (string)
- email (string, or null if not found)
- phone (string, or null if not found)
- years_experience (number, estimate from dates if not stated explicitly, else null)
- skills (list of up to 8 strings)
- education (string: highest degree + field)
- last_job_title (string or null)

CV text:
{cv_text}
"""


def extract_cv_fields(cv_text: str, client) -> dict:
    messages = [
        {"role": "system", "content": "You extract structured data from CVs. Respond only in valid JSON."},
        {"role": "user", "content": EXTRACTION_PROMPT.format(cv_text=cv_text[:6000])}
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Could not parse JSON, raw output: {raw[:200]}")
        return {
            "name": "PARSE_ERROR", "email": None, "phone": None,
            "years_experience": None, "skills": [], "education": None,
            "last_job_title": None
        }


def process_all_cvs(file_paths: list[Path] | None = None):
    pdf_files = file_paths if file_paths else list(CV_RAW_DIR.glob("*"))
    pdf_files = [path for path in pdf_files if path.is_file()]

    if not pdf_files:
        logger.warning(f"No CV PDFs found in {CV_RAW_DIR}")
        return

    client = get_openai_client()
    records = []

    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        text = extract_document_text(pdf_path)
        fields = extract_cv_fields(text, client)
        fields["source_file"] = pdf_path.stem
        records.append(fields)

    df = pd.DataFrame(records)
    df = df[["source_file", "name", "email", "phone", "years_experience",
              "last_job_title", "education", "skills"]]
    df["skills"] = df["skills"].apply(lambda s: ", ".join(s) if isinstance(s, list) else s)

    output_path = OUTPUT_DIR / "cv_shortlist.xlsx"
    df.to_excel(output_path, index=False)

    logger.info(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    process_all_cvs()
