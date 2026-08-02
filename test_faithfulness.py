"""
Example: contrast a faithful answer with an unfaithful one.

Demonstrates that the faithfulness metric penalizes answers that introduce
claims not supported by the retrieved context (here, an unsupported treatment
recommendation), even when part of the answer is correct.
"""
from rag_eval.judge import LLMJudge
from rag_eval.metrics.faithfulness import evaluate_faithfulness


CONTEXTS = [
    "A fever above 38°C in infants under 3 months is considered a "
    "medical emergency."
]

# Fully grounded: every claim is traceable to the context.
FAITHFUL_ANSWER = (
    "A fever above 38°C in an infant under 3 months is a medical emergency."
)

# Partially hallucinated: the antibiotics claim is not supported by the context.
UNFAITHFUL_ANSWER = (
    "A fever above 38°C is an emergency and antibiotics should be given "
    "immediately."
)


def main() -> None:
    judge = LLMJudge()

    print("=== Faithful answer ===")
    print(evaluate_faithfulness(judge, FAITHFUL_ANSWER, CONTEXTS))

    print("\n=== Unfaithful answer ===")
    print(evaluate_faithfulness(judge, UNFAITHFUL_ANSWER, CONTEXTS))


if __name__ == "__main__":
    main()