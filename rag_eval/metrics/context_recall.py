"""
Context recall metric.

Measures retrieval completeness against a reference answer. The ground truth
is decomposed into atomic claims, and each claim is checked for attribution
to the retrieved context. Score = attributed_claims / total_claims.

This metric is skipped (returns None) when no ground truth is available.
"""
from rag_eval.judge import LLMJudge


RECALL_SYSTEM = """You are evaluating retrieval completeness.
Given a ground-truth answer broken into claims, and a set of retrieved contexts,
decide for EACH claim whether it can be attributed to (found in) the contexts.

Return ONLY valid JSON:
{"attributions": [{"claim": "...", "attributed": "yes"}, ...]}"""


CLAIM_SYSTEM = """Break down the given answer into individual atomic claims.
Return ONLY valid JSON:
{"claims": ["claim 1", "claim 2", ...]}"""


def evaluate_context_recall(judge: LLMJudge, ground_truth: str, contexts: list) -> dict:
    """Return recall score, or None if no ground truth is supplied."""
    # Recall is undefined without a reference answer; skip rather than penalize.
    if not ground_truth:
        return {"score": None, "details": "No ground_truth provided; skipped"}

    if not contexts:
        return {"score": 0.0, "details": "No contexts provided"}

    context_text = "\n".join(contexts)

    # Step 1 — Decompose the ground truth into atomic claims.
    claim_result = judge.ask_json(
        CLAIM_SYSTEM,
        f"Answer:\n{ground_truth}",
    )
    claims = claim_result.get("claims", [])

    if not claims:
        return {"score": 0.0, "details": "No claims extracted from ground_truth"}

    # Step 2 — Check whether each claim is recoverable from the context.
    claims_list = "\n".join(f"- {c}" for c in claims)
    attr_result = judge.ask_json(
        RECALL_SYSTEM,
        f"Contexts:\n{context_text}\n\nClaims:\n{claims_list}",
    )
    attributions = attr_result.get("attributions", [])

    if not attributions:
        return {"score": 0.0, "details": "No attributions returned"}

    attributed = sum(
        1 for a in attributions if a.get("attributed", "").lower() == "yes"
    )
    score = attributed / len(attributions)

    return {
        "score": round(score, 3),
        "total_claims": len(attributions),
        "attributed_claims": attributed,
        "attributions": attributions,
    }