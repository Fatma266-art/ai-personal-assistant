def route_task(task: str) -> str:
    """Select an action from a plain-language task description."""
    task_lower = task.lower()

    cv_keywords = [
        "cv", "cvs", "candidate", "shortlist", "hire",
        "resume", "hr", "سيرة ذاتية", "مرشح", "رشح"
    ]

    report_keywords = [
        "report", "pdf report", "summarize into a report",
        "تقرير", "لخص"
    ]

    if any(keyword in task_lower for keyword in report_keywords):
        return "pdf_report"

    if any(keyword in task_lower for keyword in cv_keywords):
        return "hr_shortlist"

    return "rag"
