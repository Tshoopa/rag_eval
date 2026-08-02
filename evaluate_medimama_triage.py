"""
MediMama triage evaluation harness.

Evaluates the triage-level accuracy and API safety behavior of the MediMama
pediatric assistant against a labeled dataset.

Metrics:
    - Exact-match accuracy
    - Under-triage rate   (predicted less urgent than expected — the dangerous case)
    - Over-triage rate    (predicted more urgent than expected — the safe-but-costly case)
    - Critical emergency recall
    - API failure rate
    - Refusal rate

Triage level contract:
    Level 1 = most urgent
    Level 5 = least urgent
"""

import json
import os
from collections import Counter
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


MEDIMAMA_API_URL = os.getenv(
    "MEDIMAMA_API_URL",
    "https://fraying-flattered-luminous.ngrok-free.dev/ask",
)

DATASET_PATH = os.getenv(
    "MEDIMAMA_TRIAGE_DATASET",
    "tests/medimama_triage_cases_balanced.json",
)

REPORT_PATH = "reports/medimama_triage_results_2.json"

# Set to a small number for smoke tests; set to None to run the full dataset.
TEST_LIMIT = 100

DEFAULT_LANGUAGE = "en"
REQUEST_TIMEOUT = 90


def load_triage_dataset(path: str) -> list[dict[str, Any]]:
    """
    Load the triage dataset from JSON or JSONL.

    Accepts either a {"test_cases": [...]} object, a bare JSON array,
    or newline-delimited JSON objects (JSONL).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at:\n{path}")

    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        raise ValueError("Dataset file is empty.")

    # Standard JSON (object or array).
    if content.startswith("{") or content.startswith("["):
        parsed = json.loads(content)

        if isinstance(parsed, dict):
            test_cases = parsed.get("test_cases")
            if not isinstance(test_cases, list):
                raise ValueError("JSON object must contain a 'test_cases' array.")
            return test_cases

        if isinstance(parsed, list):
            return parsed

        raise ValueError("Unsupported JSON structure for dataset.")

    # JSONL fallback: one JSON object per line.
    test_cases = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            test_cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

    return test_cases


def validate_test_case(test_case: dict[str, Any], index: int) -> None:
    """Validate the required fields and value ranges of a single test case."""
    required_fields = ["id", "input", "child_age_months", "expected_level"]

    missing_fields = [f for f in required_fields if f not in test_case]
    if missing_fields:
        raise ValueError(
            f"Sample #{index} is missing fields: {', '.join(missing_fields)}"
        )

    # Note: bool is a subclass of int in Python, so it is rejected explicitly.
    expected_level = test_case["expected_level"]
    if (
        not isinstance(expected_level, int)
        or isinstance(expected_level, bool)
        or not 1 <= expected_level <= 5
    ):
        raise ValueError(
            f"expected_level for '{test_case.get('id')}' must be an int in [1, 5]."
        )

    child_age_months = test_case["child_age_months"]
    if (
        not isinstance(child_age_months, int)
        or isinstance(child_age_months, bool)
        or child_age_months < 0
    ):
        raise ValueError(
            f"child_age_months for '{test_case.get('id')}' must be a non-negative int."
        )

    symptoms = test_case["input"]
    if not isinstance(symptoms, str) or not symptoms.strip():
        raise ValueError(f"input for '{test_case.get('id')}' is empty or invalid.")


def extract_contexts(citations: Any) -> list[str]:
    """Extract chunk text from MediMama's citations payload."""
    contexts: list[str] = []

    if not isinstance(citations, list):
        return contexts

    for citation in citations:
        if isinstance(citation, dict):
            chunk = citation.get("chunk")
            if isinstance(chunk, str) and chunk.strip():
                contexts.append(chunk.strip())
        elif isinstance(citation, str) and citation.strip():
            contexts.append(citation.strip())

    return contexts


def query_medimama(
    symptoms: str,
    child_age_months: int,
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """
    Call the /ask endpoint and return a normalized result dict.

    On success, 'predicted_level' is guaranteed to be an int in [1, 5];
    any protocol or transport failure is reported as {"success": False, ...}.
    """
    payload = {
        "symptoms": symptoms,
        "child_age_months": child_age_months,
        "language": language,
    }

    try:
        response = requests.post(
            MEDIMAMA_API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                # Skip the ngrok interstitial page in tunneled environments.
                "ngrok-skip-browser-warning": "true",
            },
        )

        if response.status_code != 200:
            print(f"   HTTP {response.status_code}: {response.text[:1000]}")
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:2000],
                "request_payload": payload,
            }

        try:
            response_json = response.json()
        except ValueError:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": "API response was not valid JSON.",
                "raw_response": response.text[:2000],
                "request_payload": payload,
            }

        # Validate the triage level up front so downstream metrics can trust it.
        predicted_level = response_json.get("emergency_level")
        if (
            not isinstance(predicted_level, int)
            or isinstance(predicted_level, bool)
            or not 1 <= predicted_level <= 5
        ):
            return {
                "success": False,
                "status_code": response.status_code,
                "error": "Missing or invalid 'emergency_level' (expected int in [1, 5]).",
                "raw_response": response_json,
                "request_payload": payload,
            }

        citations = response_json.get("citations", [])

        return {
            "success": True,
            "status_code": response.status_code,
            "request_payload": payload,
            "answer": response_json.get("answer", ""),
            "predicted_level": predicted_level,
            "emergency_label": response_json.get("emergency_label", ""),
            "see_doctor_urgency": response_json.get("see_doctor_urgency", ""),
            "verified": bool(response_json.get("verified", False)),
            "refusal": bool(response_json.get("refusal", False)),
            "citations": citations,
            "contexts": extract_contexts(citations),
            "raw_response": response_json,
        }

    except requests.Timeout:
        return {
            "success": False,
            "status_code": None,
            "error": f"Request timed out after {REQUEST_TIMEOUT}s.",
            "request_payload": payload,
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "status_code": None,
            "error": f"Network error: {exc}",
            "request_payload": payload,
        }

    except Exception as exc:
        return {
            "success": False,
            "status_code": None,
            "error": f"Unexpected error: {exc}",
            "request_payload": payload,
        }


def classify_triage_result(expected_level: int, predicted_level: int) -> str:
    """
    Classify a prediction relative to the expected level.

    Since Level 1 is most urgent, a *higher* predicted level means the system
    judged the case less urgent than it should have — i.e. under-triage.
    """
    if predicted_level == expected_level:
        return "exact_match"
    if predicted_level > expected_level:
        return "under_triage"
    return "over_triage"


def safe_rate(numerator: int, denominator: int) -> float:
    """
    Return the rate multiplied by 100 to yield percentage points [0.0, 100.0].
    
    This fixes the dashboard display bug where rates were rendered as decimals
    and appended with a raw '%' sign (e.g., displaying 0.17% instead of 17.39%).
    """
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def build_confusion_matrix(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """
    Build a 5x5 confusion matrix over triage levels.

    Rows are the expected level, columns are the predicted level.
    Only successful, in-range results contribute to the matrix.
    """
    matrix = {
        str(expected): {str(predicted): 0 for predicted in range(1, 6)}
        for expected in range(1, 6)
    }

    for result in results:
        if not result.get("api_success"):
            continue

        expected_level = result.get("expected_level")
        predicted_level = result.get("predicted_level")

        expected_is_valid = (
            isinstance(expected_level, int)
            and not isinstance(expected_level, bool)
            and 1 <= expected_level <= 5
        )
        predicted_is_valid = (
            isinstance(predicted_level, int)
            and not isinstance(predicted_level, bool)
            and 1 <= predicted_level <= 5
        )

        if not expected_is_valid or not predicted_is_valid:
            continue

        matrix[str(expected_level)][str(predicted_level)] += 1

    return matrix


def calculate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case results into dataset-level safety and accuracy metrics."""
    total_cases = len(results)

    successful_results = [r for r in results if r.get("api_success")]
    successful_cases = len(successful_results)
    failed_cases = total_cases - successful_cases

    exact_matches = sum(
        r.get("triage_classification") == "exact_match"
        for r in successful_results
    )
    under_triage_count = sum(
        r.get("triage_classification") == "under_triage"
        for r in successful_results
    )
    over_triage_count = sum(
        r.get("triage_classification") == "over_triage"
        for r in successful_results
    )
    refusal_count = sum(bool(r.get("refusal")) for r in successful_results)
    verified_count = sum(bool(r.get("verified")) for r in successful_results)

    # Critical recall: of all true Level-1 cases, how many did we catch as Level 1?
    critical_cases = [
        r for r in successful_results if r.get("expected_level") == 1
    ]
    correctly_detected_critical = sum(
        r.get("predicted_level") == 1 for r in critical_cases
    )

    expected_level_distribution = Counter(
        r.get("expected_level") for r in results
    )
    predicted_level_distribution = Counter(
        r.get("predicted_level") for r in successful_results
    )

    # Per-category breakdown for error analysis.
    category_statistics: dict[str, dict[str, Any]] = {}
    for result in results:
        category = result.get("category", "unknown")

        if category not in category_statistics:
            category_statistics[category] = {
                "total": 0,
                "successful": 0,
                "exact_match": 0,
                "under_triage": 0,
                "over_triage": 0,
                "api_failure": 0,
            }

        stats = category_statistics[category]
        stats["total"] += 1

        if not result.get("api_success"):
            stats["api_failure"] += 1
            continue

        stats["successful"] += 1
        classification = result.get("triage_classification")
        if classification in stats:
            stats[classification] += 1

    # Add category-level rates explicitly for dashboard rendering.
    for stats in category_statistics.values():
        successful = stats["successful"]
        stats["exact_match_rate"] = safe_rate(stats["exact_match"], successful)
        stats["under_triage_rate"] = safe_rate(stats["under_triage"], successful)
        stats["over_triage_rate"] = safe_rate(stats["over_triage"], successful)

    # Build the matrix once, over all successful results.
    confusion_matrix = build_confusion_matrix(successful_results)

    # Calculate rates (now natively returned as percentages, e.g., 69.56 instead of 0.6956)
    exact_match_accuracy = safe_rate(exact_matches, successful_cases)
    under_triage_rate = safe_rate(under_triage_count, successful_cases)
    over_triage_rate = safe_rate(over_triage_count, successful_cases)
    critical_emergency_recall = safe_rate(correctly_detected_critical, len(critical_cases))
    api_failure_rate = safe_rate(failed_cases, total_cases)
    refusal_rate = safe_rate(refusal_count, successful_cases)
    verified_rate = safe_rate(verified_count, successful_cases)
    
    # Clinical Safety Rate represents the proportion of decisions that did NOT under-triage.
    clinical_safety_rate = round(100.0 - under_triage_rate, 4)

    return {
        "total_cases": total_cases,
        "successful_api_responses": successful_cases,
        "failed_api_responses": failed_cases,

        # Triage rates are computed only over valid API responses (now in % format)
        "exact_match_accuracy": exact_match_accuracy,
        "under_triage_rate": under_triage_rate,
        "over_triage_rate": over_triage_rate,
        "clinical_safety_rate": clinical_safety_rate,

        "exact_match_count": exact_matches,
        "under_triage_count": under_triage_count,
        "over_triage_count": over_triage_count,

        "critical_case_count": len(critical_cases),
        "critical_emergency_recall": critical_emergency_recall,

        "api_failure_rate": api_failure_rate,
        "refusal_rate": refusal_rate,
        "verified_rate": verified_rate,

        "expected_level_distribution": {
            str(k): v
            for k, v in sorted(expected_level_distribution.items())
        },
        "predicted_level_distribution": {
            str(k): v
            for k, v in sorted(predicted_level_distribution.items())
        },
        "confusion_matrix": confusion_matrix,
        "category_statistics": category_statistics,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary and confusion matrix to stdout."""
    print("\n" + "=" * 64)
    print("MediMama Triage Evaluation Summary")
    print("=" * 64)

    print(f"Total cases:              {summary['total_cases']}")
    print(f"Successful API responses: {summary['successful_api_responses']}")
    print(f"API failures:             {summary['failed_api_responses']}")
    print(f"Exact-match accuracy:     {summary['exact_match_accuracy']:.2f}%")
    print(f"Under-triage rate:        {summary['under_triage_rate']:.2f}%")
    print(f"Over-triage rate:         {summary['over_triage_rate']:.2f}%")
    print(f"Clinical safety rate:     {summary['clinical_safety_rate']:.2f}%")
    print(f"Critical recall:          {summary['critical_emergency_recall']:.2f}%")
    print(f"API failure rate:         {summary['api_failure_rate']:.2f}%")
    print(f"Refusal rate:             {summary['refusal_rate']:.2f}%")
    print(f"Verified rate:            {summary['verified_rate']:.2f}%")

    print("\nConfusion Matrix (rows = expected, columns = predicted)")
    print("-" * 44)
    print("Exp\\Pred |  1 |  2 |  3 |  4 |  5")
    print("-" * 44)

    matrix = summary["confusion_matrix"]
    for expected_level in range(1, 6):
        row = matrix[str(expected_level)]
        print(
            f"{expected_level:^8} | "
            f"{row['1']:>2} | {row['2']:>2} | {row['3']:>2} | "
            f"{row['4']:>2} | {row['5']:>2}"
        )

    print("-" * 44)
    print("=" * 64)

    # Surface safety-critical conditions loudly.
    if summary["under_triage_count"] > 0:
        print(f"WARNING: {summary['under_triage_count']} under-triage cases detected.")
    if summary["api_failure_rate"] > 0:
        print("WARNING: some requests failed at the API layer.")


def main() -> None:
    print("Starting MediMama triage evaluation")
    print(f"API:     {MEDIMAMA_API_URL}")
    print(f"Dataset: {DATASET_PATH}\n")

    try:
        test_cases = load_triage_dataset(DATASET_PATH)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return

    selected_cases = (
        test_cases[:TEST_LIMIT] if TEST_LIMIT is not None else test_cases
    )

    print(f"Total cases in dataset: {len(test_cases)}")
    print(f"Selected for this run:  {len(selected_cases)}\n")

    results: list[dict[str, Any]] = []

    for index, test_case in enumerate(selected_cases, start=1):
        try:
            validate_test_case(test_case, index)
        except ValueError as exc:
            print(f"Skipping invalid case: {exc}")
            results.append({
                "id": test_case.get("id", f"invalid-{index}"),
                "category": test_case.get("category", "unknown"),
                "expected_level": test_case.get("expected_level"),
                "api_success": False,
                "error": str(exc),
            })
            continue

        case_id = test_case["id"]
        symptoms = test_case["input"]
        child_age_months = test_case["child_age_months"]
        expected_level = test_case["expected_level"]
        language = test_case.get("language", DEFAULT_LANGUAGE)

        print(
            f"[{index}/{len(selected_cases)}] "
            f"{case_id} | {test_case.get('category', 'unknown')}"
        )
        print(f"   Age: {child_age_months} months | Expected level: {expected_level}")
        print(f"   Input: {symptoms[:100]}...")

        api_result = query_medimama(
            symptoms=symptoms,
            child_age_months=child_age_months,
            language=language,
        )

        result: dict[str, Any] = {
            "id": case_id,
            "category": test_case.get("category", "unknown"),
            "input": symptoms,
            "child_age_months": child_age_months,
            "language": language,
            "expected_level": expected_level,
            "expected_concept": test_case.get("expected_concept"),
            "rationale": test_case.get("rationale"),
            "api_success": api_result.get("success", False),
        }

        if not api_result.get("success"):
            result.update({
                "error": api_result.get("error"),
                "status_code": api_result.get("status_code"),
                "request_payload": api_result.get("request_payload"),
                "raw_response": api_result.get("raw_response"),
            })
            results.append(result)
            print(f"   Request failed: {api_result.get('error')}\n")
            continue

        predicted_level = api_result["predicted_level"]
        classification = classify_triage_result(
            expected_level=expected_level,
            predicted_level=predicted_level,
        )

        result.update({
            "predicted_level": predicted_level,
            "triage_classification": classification,
            "is_exact_match": classification == "exact_match",
            "is_under_triage": classification == "under_triage",
            "is_over_triage": classification == "over_triage",
            "answer": api_result.get("answer"),
            "emergency_label": api_result.get("emergency_label"),
            "see_doctor_urgency": api_result.get("see_doctor_urgency"),
            "verified": api_result.get("verified"),
            "refusal": api_result.get("refusal"),
            "contexts": api_result.get("contexts"),
            "citations": api_result.get("citations"),
            "raw_response": api_result.get("raw_response"),
        })
        results.append(result)

        status_icon = {
            "exact_match": "[OK]",
            "under_triage": "[UNDER]",
            "over_triage": "[OVER]",
        }.get(classification, "[?]")

        print(f"   {status_icon} Predicted: {predicted_level} | {classification}")
        print(f"   Citations: {len(api_result.get('contexts', []))}\n")

    summary = calculate_summary(results)

    output = {
        "metadata": {
            "api_url": MEDIMAMA_API_URL,
            "dataset_path": DATASET_PATH,
            "level_contract": {"1": "most urgent", "5": "least urgent"},
            "test_limit": TEST_LIMIT,
        },
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nRaw results and safety metadata saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()