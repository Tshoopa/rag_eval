"""
Answer relevance metric.

Measures how directly an answer addresses the original question.
A single LLM judge call rates the answer on topicality and focus,
returning a normalized score in the [0, 1] range.
"""
from rag_eval.judge import LLMJudge


RELEVANCE_SYSTEM = """You are an evaluator of answer relevance.
Given a question and an answer, rate how directly the answer addresses the question.
Consider:
- Does the answer stay on topic?
- Does it avoid irrelevant information?
- Does it actually answer what was asked?

Return ONLY valid JSON:
{"relevance_score": <float between 0 and 1>, "reasoning": "brief explanation"}"""


def evaluate_answer_relevance(judge: LLMJudge, question: str, answer: str) -> dict:
    """Return a relevance score in [0, 1] with the judge's reasoning."""
    result = judge.ask_json(
        RELEVANCE_SYSTEM,
        f"Question:\n{question}\n\nAnswer:\n{answer}",
    )

    # Coerce and clamp the score; the LLM occasionally returns strings
    # or out-of-range values.
    score = result.get("relevance_score", 0.0)
    try:
        score = float(score)
        score = max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        score = 0.0

    return {
        "score": round(score, 3),
        "reasoning": result.get("reasoning", ""),
    }