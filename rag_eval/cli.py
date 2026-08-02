"""
Command-line interface for RAG-Eval.

Usage:
    python -m rag_eval --dataset <path> --output <path>
"""
import click
import json
import os

from rag_eval.evaluator import RAGEvaluator, load_dataset
from rag_eval.judge import LLMJudge
from rag_eval.report import generate_html_report


@click.command()
@click.option(
    "--dataset", "-d",
    required=True,
    help="Path to the evaluation dataset (JSON).",
)
@click.option(
    "--output", "-o",
    default="reports/report.html",
    help="Output path for the HTML report.",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Judge model name (defaults to DEEPSEEK_MODEL from .env).",
)
@click.option(
    "--json-output", "-j",
    default=None,
    help="Optional path to also dump raw results as JSON.",
)
def main(dataset, output, model, json_output):
    """RAG-Eval — evaluate RAG systems using LLM-as-a-judge."""

    if not os.path.exists(dataset):
        click.echo(f"Dataset not found: {dataset}")
        return

    click.echo(f"Loading dataset from: {dataset}")
    data = load_dataset(dataset)
    click.echo(f"  Loaded {len(data)} samples.\n")

    judge = LLMJudge(model=model)
    evaluator = RAGEvaluator(judge)
    results = evaluator.evaluate_dataset(data)

    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        click.echo(f"Raw results saved to: {json_output}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    generate_html_report(results, output)
    click.echo(f"\nHTML report generated: {output}")
    click.echo("  Open it in a browser to view.")


if __name__ == "__main__":
    main()