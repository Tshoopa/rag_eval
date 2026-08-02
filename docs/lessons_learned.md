# Lessons Learned: The Challenge of "Implicit Entailment" in LLM-as-a-Judge

When building `RAG_Eval` to evaluate my medical RAG system, I utilized the **LLM-as-a-Judge** pattern using DeepSeek. While testing the **Faithfulness** and **Context Recall** metrics, I uncovered a critical challenge in how LLMs evaluate clinical texts.

## The Scenario
I provided the judge with the following test case:

*   **Retrieved Context:** "Fever in infants under 3 months with a temperature over 38°C is considered an emergency."
*   **System's Answer:** "Yes, a fever over 38°C in an infant under 3 months is an emergency and requires an immediate doctor's visit."

## The Unexpected Result
The LLM Judge gave this answer a **0.5 / 1.0 score** for Faithfulness. Why? 
It broke the answer down into two claims:
1. "It is an emergency" ➔ **Supported (Yes)**
2. "Requires an immediate doctor's visit" ➔ **Not Supported (No)**

## The Engineering Takeaway
The LLM Judge was being overly strict. In human medical reasoning, if a condition is "considered an emergency," it *implicitly entails* that it "requires an immediate doctor's visit." However, the LLM failed to grasp this **implicit entailment** because it was looking for an exact semantic match.

**How to solve this?**
This taught me that simply throwing a prompt at an LLM is not enough for evaluation. To fix this, I realized that:
1. **Prompt Engineering is crucial:** The system prompt for the Judge must explicitly instruct it to account for reasonable implicit logical steps (like Emergency = Immediate Care).
2. **Threshold Tuning:** In medical RAG evaluation, you cannot rely purely on binary (Yes/No) automated scoring without human-in-the-loop validation for edge cases.

This experience shifted my perspective from just being a "RAG builder" to an "AI Safety Evaluator," understanding the deep nuances of LLM reasoning limitations.