from pathlib import Path


def _as_paths(file_path: str | list[str] | None) -> list[Path]:
    if not file_path:
        return []
    return [Path(path) for path in file_path] if isinstance(file_path, list) else [Path(file_path)]


def run_rag_tool(task: str, file_path: str | list[str] | None = None) -> dict:
    from src.llm.reason import reason
    from src.ingestion.ingest_single_file import ingest_file
    from src.retrieval.retrieve import load_retriever

    try:
        for path in _as_paths(file_path):
            chunks_added = ingest_file(path)
            if chunks_added == 0:
                return {"status": "error", "message": f"Could not extract text from {path.name}.", "output_file": None}
        if file_path:
            load_retriever.cache_clear()

        result = reason(task)
        sources = list(dict.fromkeys(
            f"{item['source']} — page {item['page']}" for item in result["retrieved_chunks"]
        ))
        return {
            "status": "success",
            "message": result["answer"],
            "output_file": None,
            "sources": sources,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"RAG tool failed: {e}",
            "output_file": None
        }

def run_hr_shortlist_tool(
    task: str,
    file_path: str = None,
    job_title: str | None = None,
    required_skills: list[str] | None = None,
    shortlist_threshold: int | None = None,
) -> dict:
    from src.actions.hr_shortlist_scored import score_candidates, OUTPUT_DIR

    try:
        files = _as_paths(file_path) or None
        dataframe = score_candidates(files, job_title, required_skills, shortlist_threshold)
        import re
        safe_title = re.sub(r"[^a-z0-9]+", "_", (job_title or "Junior Data Analyst").lower()).strip("_") or "role"
        output_path = OUTPUT_DIR / f"shortlist_{safe_title}.xlsx"
        return {
            "status": "success",
            "message": "HR shortlisting completed.",
            "output_file": str(output_path),
            "preview": dataframe.head(10).to_dict(orient="records") if dataframe is not None else [],
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"HR shortlist tool failed: {e}",
            "output_file": None
        }


def run_pdf_report_tool(task: str, file_path: str = None) -> dict:
    from src.actions.pdf_report import generate_report
    from pathlib import Path

    if not file_path:
        return {
            "status": "error",
            "message": "This task needs a file to generate a report from. Please upload one.",
            "output_file": None
        }

    try:
        output_path = generate_report(Path(file_path), task)
        return {
            "status": "success",
            "message": "Report generated successfully.",
            "output_file": str(output_path)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"PDF report tool failed: {e}",
            "output_file": None
        }    
