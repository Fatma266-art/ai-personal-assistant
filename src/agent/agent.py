from src.agent.router import route_task
from src.agent.tools import run_hr_shortlist_tool, run_pdf_report_tool, run_rag_tool


TOOL_MAP = {
    "rag": run_rag_tool,
    "hr_shortlist": run_hr_shortlist_tool,
    "pdf_report": run_pdf_report_tool,
}


def run_agent(
    task: str,
    file_path: str | None = None,
    job_title: str | None = None,
    required_skills: list[str] | None = None,
    shortlist_threshold: int | None = None,
) -> dict:
    """Route a task to the appropriate tool and return its result."""
    route = route_task(task)
    tool = TOOL_MAP.get(route)

    if tool is None:
        return {"status": "error", "message": "Unknown task type.", "output_file": None}

    if route == "hr_shortlist":
        return tool(task, file_path, job_title, required_skills, shortlist_threshold)
    if route == "pdf_report" and isinstance(file_path, list):
        return tool(task, file_path[0] if file_path else None)
    return tool(task, file_path)
