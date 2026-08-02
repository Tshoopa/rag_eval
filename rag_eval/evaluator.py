"""
Evaluation orchestrator.

Runs all four RAG metrics over a dataset and aggregates the results into
per-sample scores plus dataset-level averages.
"""
import json
from statistics import mean

from rag_eval.judge import LLMJudge
from rag_eval.metrics.faithfulness import evaluate_faithfulness
from rag_eval.metrics.answer_relevance import evaluate_answer_relevance
from rag_eval.metrics.context_precision import evaluate_context_precision
from rag_eval.metrics.context_recall import evaluate_context_recall


class RAGEvaluator:
    """Runs the full metric suite over a dataset."""

    def __init__(self, judge: LLMJudge = None):
        self.judge = judge or LLMJudge()

    def evaluate_sample(self, sample: dict, index: int = 0) -> dict:
        """Evaluate a single sample across all four metrics."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        contexts = sample.get("contexts", [])
        ground_truth = sample.get("ground_truth", "")

        print(f"  -> Evaluating sample {index + 1}...")

        faithfulness = evaluate_faithfulness(self.judge, answer, contexts)
        relevance = evaluate_answer_relevance(self.judge, question, answer)
        precision = evaluate_context_precision(self.judge, question, contexts)
        recall = evaluate_context_recall(self.judge, ground_truth, contexts)

        return {
            "question": question,
            "answer": answer,
            "scores": {
                "faithfulness": faithfulness.get("score"),
                "answer_relevance": relevance.get("score"),
                "context_precision": precision.get("score"),
                "context_recall": recall.get("score"),
            },
            "details": {
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
                "context_precision": precision,
                "context_recall": recall,
            },
        }

    def evaluate_dataset(self, dataset: list) -> dict:
        """Evaluate every sample and return per-sample results plus aggregates."""
        print(f"\nStarting evaluation of {len(dataset)} samples...\n")

        results = [
            self.evaluate_sample(sample, i)
            for i, sample in enumerate(dataset)
        ]

        aggregate = self._compute_aggregate(results)

        print("\nEvaluation complete.\n")
        return {
            "results": results,
            "aggregate": aggregate,
            "total_samples": len(dataset),
        }

    @staticmethod
    def _compute_aggregate(results: list) -> dict:
        """Average each metric across samples, ignoring skipped (None) scores."""
        metrics = [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
        ]
        aggregate = {}

        for metric in metrics:
            valid_scores = [
                r["scores"][metric]
                for r in results
                if r["scores"][metric] is not None
            ]
            aggregate[metric] = (
                round(mean(valid_scores), 3) if valid_scores else None
            )

        return aggregate


def load_dataset(path: str) -> list:
    """Load an evaluation dataset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_from_file(dataset_path: str, judge: LLMJudge = None) -> dict:
    """Convenience helper: load a dataset from disk and evaluate it."""
    dataset = load_dataset(dataset_path)
    evaluator = RAGEvaluator(judge)
    return evaluator.evaluate_dataset(dataset)