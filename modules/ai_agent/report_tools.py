def generate_report(report_content_markdown: str, base_dir: str) -> dict:
    """Saves the final generated markdown report to the base_dir."""
    try:
        from pathlib import Path
        p = Path(base_dir) / "ai_agent_report.md"
        with open(p, "w", encoding="utf-8") as f:
            f.write(report_content_markdown)
        return {"status": "success", "report_path": str(p)}
    except Exception as e:
        return {"error": str(e)}
