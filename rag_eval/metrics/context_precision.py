"""
Context precision metric.

Measures retrieval signal-to-noise: of all retrieved chunks, how many are
actually relevant to the question. Each chunk is judged independently.
Score = relevant_chunks / total_chunks.
"""
from rag_eval.judge import LLMJudge


PRECISION_SYSTEM = """You are evaluating retrieval quality.
Given a question and a list of retrieved context chunks, decide for EACH chunk
whether it is relevant and useful for answering the question.

Return ONLY valid JSON:
{"relevance": [{"chunk_index": 0, "relevant": "yes"}, {"chunk_index": 1, "relevant": "no"}, ...]}"""


def evaluate_context_precision(judge: LLMJudge, question: str, contexts: list) -> dict:
    """Return the fraction of retrieved chunks judged relevant."""
    if not contexts:
        return {"score": 0.0, "details": "No contexts provided"}

    # Index chunks so the judge can reference them unambiguously.
    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))

    result = judge.ask_json(
        PRECISION_SYSTEM,
        f"Question:\n{question}\n\nRetrieved chunks:\n{numbered}",
    )
    relevance = result.get("relevance", [])

    if not relevance:
        return {"score": 0.0, "details": "No relevance verdicts returned"}

    relevant_count = sum(
        1 for r in relevance if r.get("relevant", "").lower() == "yes"
    )
    score = relevant_count / len(relevance)

    return {
        "score": round(score, 3),
        "total_chunks": len(relevance),
        "relevant_chunks": relevant_count,
        "relevance": relevance,
    }