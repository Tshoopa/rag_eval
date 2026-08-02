"""
Example: generate an HTML report from a saved evaluation results file.

Reads the JSON produced by the evaluator and renders a standalone HTML report.

Note: this expects the standard RAG-Eval output schema
({"aggregate": ..., "results": ..., "total_samples": ...}). MediMama triage
output uses a different schema ({"summary": ..., "results": ...}) and should
be rendered with the dedicated MediMama report generator instead.
"""
import json

from rag_eval.report import generate_html_report


INPUT_PATH = "eval_results.json"
OUTPUT_PATH = "reports/report.html"


def main() -> None:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        eval_output = json.load(f)

    generate_html_report(eval_output, OUTPUT_PATH)
    print(f"Report generated. Open {OUTPUT_PATH} in a browser.")


if __name__ == "__main__":
    main()