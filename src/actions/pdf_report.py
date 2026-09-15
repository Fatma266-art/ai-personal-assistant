"""
PDF Report Generation
----------------------
Takes an uploaded document, asks the LLM to structure its content
into a report (title + sections), and renders that into a formatted PDF.
"""

from pathlib import Path
import json
import logging
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from src.ingestion.extract_text import extract_text
from src.llm.reason import get_openai_client, LLM_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "Data" / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


REPORT_PROMPT = """You will structure the following document into a clear report.

Return ONLY valid JSON, no explanation, no markdown fences.

JSON shape:
{{
  "title": "string - a short, descriptive report title",
  "sections": [
    {{"heading": "string", "body": "string - a few sentences, plain text, no markdown"}}
  ]
}}

Use 3 to 6 sections. Base everything only on the document content below.

User's request about this document: {task}

Document content:
{content}
"""


def build_report_data(content: str, task: str, client) -> dict:
    messages = [
        {"role": "system", "content": "You structure documents into clear, well-organized reports. Respond only in valid JSON."},
        {"role": "user", "content": REPORT_PROMPT.format(task=task, content=content[:8000])}
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Could not parse JSON, raw output: {raw[:200]}")
        return {
            "title": "Report",
            "sections": [{"heading": "Content", "body": raw}]
        }


def render_pdf(report_data: dict, output_path: Path) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                             topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "SectionBody", parent=styles["Normal"], leading=16
    )

    story = [
        Paragraph(report_data.get("title", "Report"), styles["Title"]),
        Spacer(1, 16)
    ]

    for section in report_data.get("sections", []):
        story.append(Paragraph(section.get("heading", ""), heading_style))
        story.append(Paragraph(section.get("body", ""), body_style))

    doc.build(story)


def generate_report(file_path: Path, task: str) -> Path:
    """
    Main entry point: PDF in -> structured PDF report out.
    Returns the path to the generated report.
    """
    logger.info(f"Generating report from: {file_path.name}")

    text = extract_text(file_path)

    if not text.strip():
        raise ValueError("Could not extract any text from the uploaded file.")

    client = get_openai_client()
    report_data = build_report_data(text, task, client)

    output_path = OUTPUT_DIR / f"report_{file_path.stem}.pdf"
    render_pdf(report_data, output_path)

    logger.info(f"✅ Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    test_file = Path(input("PDF path: ").strip())
    test_task = input("What should the report focus on? ").strip()
    generate_report(test_file, test_task)
