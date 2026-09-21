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

from xml.sax.saxutils import escape
import re

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

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
Write the report in the same language as the user's request below.

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


# =========================
# Arabic support (fonts + RTL text shaping)
# =========================
# ReportLab's built-in fonts have no Arabic glyphs (they show as black boxes),
# and ReportLab does not shape or reorder RTL text by itself. So we:
#   1) register a TTF font that contains Arabic (Amiri, OFL license), and
#   2) reshape the letters + apply bidi ordering line by line.

FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
pdfmetrics.registerFont(TTFont("Amiri", str(FONT_DIR / "Amiri-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Amiri-Bold", str(FONT_DIR / "Amiri-Bold.ttf")))

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def to_visual(line: str) -> str:
    """Logical Arabic text -> what should be drawn left-to-right on the page."""
    return get_display(arabic_reshaper.reshape(line))


def wrap_rtl(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Break text into lines that fit max_width, measuring the shaped text."""
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and pdfmetrics.stringWidth(to_visual(trial), font, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def add_block(story: list, text: str, latin_style, arabic_style, max_width: float) -> None:
    text = (text or "").strip()
    if not text:
        return
    if not has_arabic(text):
        story.append(Paragraph(escape(text), latin_style))
        return
    for line in wrap_rtl(text, arabic_style.fontName, arabic_style.fontSize, max_width):
        story.append(Paragraph(escape(to_visual(line)), arabic_style))


def render_pdf(report_data: dict, output_path: Path) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                             topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    styles = getSampleStyleSheet()
    max_width = letter[0] - doc.leftMargin - doc.rightMargin

    # English styles (unchanged)
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "SectionBody", parent=styles["Normal"], leading=16
    )

    # Arabic styles: Amiri font, right-aligned
    ar_title = ParagraphStyle(
        "ArTitle", parent=styles["Title"], fontName="Amiri-Bold",
        fontSize=22, leading=30, alignment=TA_CENTER
    )
    ar_heading = ParagraphStyle(
        "ArHeading", parent=styles["Heading2"], fontName="Amiri-Bold",
        fontSize=16, leading=24, alignment=TA_RIGHT, spaceBefore=14, spaceAfter=4
    )
    ar_body = ParagraphStyle(
        "ArBody", parent=styles["Normal"], fontName="Amiri",
        fontSize=13, leading=22, alignment=TA_RIGHT
    )

    story = []
    add_block(story, report_data.get("title", "Report"), styles["Title"], ar_title, max_width)
    story.append(Spacer(1, 16))

    for section in report_data.get("sections", []):
        add_block(story, section.get("heading", ""), heading_style, ar_heading, max_width)
        add_block(story, section.get("body", ""), body_style, ar_body, max_width)

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
