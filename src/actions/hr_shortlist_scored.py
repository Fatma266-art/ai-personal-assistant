from pathlib import Path
import logging
import sys
import re

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.llm.reason import get_openai_client
from src.actions.hr_shortlist import CV_RAW_DIR, OUTPUT_DIR, extract_document_text, extract_cv_fields
from src.ingestion.extract_text import SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_JOB_TITLE = "Junior Data Analyst"
DEFAULT_REQUIRED_SKILLS = ["Python", "SQL", "Power BI", "Excel"]
DEFAULT_SHORTLIST_THRESHOLD = 50


def match_skills(candidate_skills, required_skills):
    candidate_text = " | ".join(
        s for s in candidate_skills if isinstance(s, str)
    ).lower()

    matched, missing = [], []
    for required in required_skills:
        pattern = r"\b" + re.escape(required.lower()) + r"\b"
        if re.search(pattern, candidate_text):
            matched.append(required)
        else:
            missing.append(required)

    return matched, missing


def calculate_score(matched_skills, required_skills):
    if not required_skills:
        return 0
    return round(len(matched_skills) / len(required_skills) * 100)


def score_candidates(
    file_paths: list[Path] | None = None,
    job_title: str | None = None,
    required_skills: list[str] | None = None,
    shortlist_threshold: int | None = None,
):
    custom_role = bool(job_title and job_title.strip())
    job_title = (job_title or DEFAULT_JOB_TITLE).strip()
    if custom_role and not required_skills:
        raise ValueError("Enter the required skills for the role so candidates can be scored correctly.")
    required_skills = [skill.strip() for skill in (required_skills or DEFAULT_REQUIRED_SKILLS) if skill.strip()]
    shortlist_threshold = DEFAULT_SHORTLIST_THRESHOLD if shortlist_threshold is None else shortlist_threshold
    if not required_skills:
        raise ValueError("Enter at least one required skill for the role.")
    if not 0 <= shortlist_threshold <= 100:
        raise ValueError("Shortlist threshold must be between 0 and 100.")

    pdf_files = file_paths if file_paths else [
        path for path in CV_RAW_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not pdf_files:
        logger.warning(f"No supported CV documents found in {CV_RAW_DIR}")
        return

    client = get_openai_client()
    rows = []

    for pdf_path in pdf_files:
        logger.info(f"Scoring: {pdf_path.name}")
        text = extract_document_text(pdf_path)
        fields = extract_cv_fields(text, client)

        candidate_skills = fields.get("skills") or []
        matched, missing = match_skills(candidate_skills, required_skills)
        score = calculate_score(matched, required_skills)
        decision = "Shortlist" if score >= shortlist_threshold else "Reject"

        rows.append({
            "Candidate": fields.get("name"),
            "Email": fields.get("email"),
            "Phone": fields.get("phone"),
            "Experience": fields.get("years_experience"),
            "Job Title": fields.get("last_job_title"),
            "Education": fields.get("education"),
            "Role": job_title,
            "Required Skills": ", ".join(required_skills),
            "Matched Skills": ", ".join(matched) if matched else "—",
            "Missing Skills": ", ".join(missing) if missing else "—",
            "Score": score,
            "Decision": decision,
            "Source": pdf_path.stem
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Score", ascending=False)
    df["Score"] = df["Score"].apply(lambda x: f"{x}%")

    safe_title = re.sub(r"[^a-z0-9]+", "_", job_title.lower()).strip("_") or "role"
    output_path = OUTPUT_DIR / f"shortlist_{safe_title}.xlsx"
    try:
        df.to_excel(output_path, index=False)
        logger.info(f"✅ Saved: {output_path}")
        return df
    except PermissionError:
        logger.error(
            f"❌ Could not save — '{output_path.name}' is open somewhere "
            f"(probably Excel). Close it and run again."
        )


if __name__ == "__main__":
    score_candidates()
