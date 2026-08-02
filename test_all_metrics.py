"""
Example: run all four RAG metrics on a single hand-crafted sample.

Useful as a smoke test to confirm the judge is reachable and each metric
returns a well-formed result. The sample deliberately includes one
irrelevant context chunk to exercise the context-precision metric.
"""
from rag_eval.judge import LLMJudge
from rag_eval.metrics.faithfulness import evaluate_faithfulness
from rag_eval.metrics.answer_relevance import evaluate_answer_relevance
from rag_eval.metrics.context_precision import evaluate_context_precision
from rag_eval.metrics.context_recall import evaluate_context_recall


SAMPLE = {
    "question": "Is a fever of 38°C dangerous in a newborn?",
    "answer": (
        "Yes, a fever above 38°C in an infant under 3 months is an emergency "
        "and requires immediate medical attention."
    ),
    "contexts": [
        "A fever above 38°C in infants under 3 months is considered a "
        "medical emergency.",
        # Deliberately irrelevant chunk to test context precision.
        "The capital of Iran is Tehran.",
    ],
    "ground_truth": (
        "A fever above 38°C in an infant under 3 months is an emergency "
        "and requires immediate medical attention."
    ),
}


def main() -> None:
    judge = LLMJudge()

    print("=" * 50)
    print("1) Faithfulness")
    print(evaluate_faithfulness(judge, SAMPLE["answer"], SAMPLE["contexts"]))

    print("\n2) Answer Relevance")
    print(evaluate_answer_relevance(judge, SAMPLE["question"], SAMPLE["answer"]))

    print("\n3) Context Precision")
    print(evaluate_context_precision(judge, SAMPLE["question"], SAMPLE["contexts"]))

    print("\n4) Context Recall")
    print(evaluate_context_recall(judge, SAMPLE["ground_truth"], SAMPLE["contexts"]))
    print("=" * 50)


if __name__ == "__main__":
    main()