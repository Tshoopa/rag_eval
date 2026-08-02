"""
Faithfulness metric.

Measures whether an answer is grounded in the retrieved context.
The answer is decomposed into atomic claims, and each claim is verified
against the context. The final score is the fraction of claims that are
directly supported by the retrieved evidence.
"""
from rag_eval.judge import LLMJudge


CLAIM_EXTRACTION_SYSTEM = """You are a precise text analyzer.
Break down the given answer into individual, atomic factual claims.
Each claim should be a single, verifiable statement.
Return ONLY valid JSON in this format:
{"claims": ["claim 1", "claim 2", ...]}"""


VERDICT_SYSTEM = """You are a strict fact-checker.
Given a context and a claim, decide if the claim is DIRECTLY supported by the context.
Answer only "yes" if the context clearly supports the claim, otherwise "no".
Return ONLY valid JSON in this format:
{"verdicts": [{"claim": "...", "supported": "yes"}, ...]}"""


def evaluate_faithfulness(judge: LLMJudge, answer: str, contexts: list) -> dict:
    """Compute a faithfulness score: supported_claims / total_claims."""
    context_text = "\n".join(contexts)

    # Step 1 — Decompose the answer into atomic claims.
    claim_result = judge.ask_json(
        CLAIM_EXTRACTION_SYSTEM,
        f"Answer to analyze:\n{answer}",
    )
    claims = claim_result.get("claims", [])

    if not claims:
        return {"score": 0.0, "claims": [], "details": "No claims extracted"}

    # Step 2 — Verify each claim against the retrieved context.
    claims_list = "\n".join(f"- {c}" for c in claims)
    verdict_result = judge.ask_json(
        VERDICT_SYSTEM,
        f"Context:\n{context_text}\n\nClaims to verify:\n{claims_list}",
    )
    verdicts = verdict_result.get("verdicts", [])

    if not verdicts:
        return {"score": 0.0, "claims": claims, "details": "No verdicts returned"}

    # Step 3 — Score is the ratio of context-supported claims.
    supported = sum(1 for v in verdicts if v.get("supported", "").lower() == "yes")
    score = supported / len(verdicts)

    return {
        "score": round(score, 3),
        "total_claims": len(verdicts),
        "supported_claims": supported,
        "verdicts": verdicts,
    }