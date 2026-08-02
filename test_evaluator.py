"""
Example: evaluate a full dataset and persist the results.

Loads a dataset from disk, runs the complete metric suite, prints a simple
ASCII bar chart of the aggregate scores, and writes the full results to JSON
for later report generation.
"""
import json

from rag_eval.evaluator import evaluate_from_file


DATASET_PATH = "tests/sample_dataset.json"
OUTPUT_PATH = "eval_results.json"


def print_aggregate_bars(aggregate: dict) -> None:
    """Render aggregate scores as a simple terminal bar chart."""
    for metric, score in aggregate.items():
        # Scale [0, 1] scores to a 20-character bar; None renders as empty.
        bar = "#" * int((score or 0) * 20)
        print(f"  {metric:20s}: {score}  {bar}")


def main() -> None:
    result = evaluate_from_file(DATASET_PATH)

    print("=" * 50)
    print("Aggregate Scores")
    print("=" * 50)
    print_aggregate_bars(result["aggregate"])
    print(f"\n  Total samples: {result['total_samples']}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()