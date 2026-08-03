# RAG_Eval

> **An independent, LLM-as-a-Judge evaluation harness for clinical RAG systems.**

Most RAG evaluation libraries answer *"is this answer good?"*.
In healthcare, that is the wrong question. The right question is
*"can this answer hurt someone?"*

`RAG_Eval` was built to answer the second one.

> ⚠️ **Status:** Research prototype. Built and validated against a single
> pediatric triage system ([MediMama](https://github.com/Tshoopa/MediMama)).
> It is not a general-purpose benchmark suite, and not a certified medical
> evaluation tool.

---

## 🎯 Why this exists

While building a pediatric RAG system, I needed to answer a question that
existing frameworks did not cover:

**A generic RAG evaluator treats every error equally.
A clinical evaluator cannot.**

If a triage system says *"this can wait until morning"* about a case that
actually needs an emergency room, that is not a low score — that is a failure
mode with a name: a **dangerous downgrade**. It has to be counted separately,
and it has to be zero.

So `RAG_Eval` sits *outside* the target system as an independent inspector.
It does not import the RAG pipeline, does not share its embedding model, and
does not reuse its prompts. It only sees what a user would see:
**the query, the retrieved context, and the final answer.**

---

## 📊 Results on MediMama


![Evaluation Dashboard](assets/medimama_eval.png)
![Evaluation Dashboard](assets/medimama_eval2.png)
![Evaluation Dashboard](assets/medimama_evaluation.png)

| Metric | Score | Notes |
|---|---|---|
| **Clinical Safety** | `100%` | 0 dangerous downgrades across evaluated test cases. (See dashboard screenshot) |
| Faithfulness | `WIP` | *Full benchmark pending final dataset validation* |
| Answer Relevance | `WIP` | *Full benchmark pending final dataset validation* |
| Context Precision | `WIP` | *Full benchmark pending final dataset validation* |
| Context Recall | `WIP` | *Full benchmark pending final dataset validation* |

*(Note: WIP = Work in Progress. Current focus has been strictly on optimizing the Clinical Safety threshold and Triage Consistency.)*

**Evaluation set:** `__` hand-written pediatric queries with clinician-style
ground truth, covering `__` triage levels.
**Judge model:** `deepseek-chat`, temperature `0.0`.

---

## ⚙️ How it works

```mermaid
flowchart LR
    A[Eval dataset<br/>query + ground truth] --> B[Target RAG system]
    B --> C[answer + retrieved context]
    C --> D[RAG_Eval]
    D --> E[Metric prompts]
    E --> F[LLM-as-a-Judge<br/>DeepSeek API]
    F --> G[Pydantic-validated<br/>structured scores]
    G --> H[Report / dashboard]
```

Every metric is a separate, isolated judge call. Scores are returned as
structured JSON and validated with **Pydantic** before they are ever
aggregated — so a malformed judge response fails loudly instead of silently
polluting the average.

---

## 📐 Metrics

### Generation quality

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context, or did the model invent something? |
| **Answer Relevance** | Does the answer address the actual question, without drifting into generic advice? |

### Retrieval quality

| Metric | Question it answers |
|---|---|
| **Context Precision** | How much of what we retrieved was actually useful? (measures noise) |
| **Context Recall** | Did we retrieve *everything* needed to answer correctly? (measures gaps) |

### Clinical safety — the reason this project exists

| Metric | Question it answers |
|---|---|
| **Triage Consistency** | Does the predicted urgency level match the ground-truth level? |
| **Dangerous Downgrade Rate** | How often did the system predict a *lower* urgency than reality? |

The two are deliberately separate.
An **upgrade** (saying "go to the ER" when home care was enough) is a cost.
A **downgrade** is a harm. Averaging them into one accuracy number hides
exactly the failure you most need to see.

---

## 🗂️ Project structure

<!-- TODO: run `tree -L 2 -I '__pycache__|venv'` and paste your real output -->

```
rag_eval/
├── rag_eval/
│   ├── __init__.py          # public API surface
│   ├── metrics/             # one module per metric
│   ├── judge.py             # DeepSeek client + retry logic
│   └── schemas.py           # Pydantic response models
├── tests/
├── assets/
├── test_all_metrics.py      # end-to-end smoke run
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick start

**1. Clone and install**

```bash
git clone https://github.com/USERNAME/rag_eval.git
cd rag_eval
pip install -r requirements.txt
```

**2. Configure the judge**

```bash
cp .env.example .env
```

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

**3. Run the full metric suite**

```bash
python test_all_metrics.py
```

This runs every metric against the bundled sample cases and prints a score
table. It is the fastest way to confirm your API key and environment work.

---

## 🧩 Using it on your own system

`RAG_Eval` never calls your pipeline. You run your pipeline, then hand the
result over:

```python
from rag_eval import evaluate

result = evaluate(
    query="My 2-year-old has had a fever of 38.5°C since last night.",
    answer=my_rag_system.answer(...),        # your system's output
    contexts=my_rag_system.retrieved_docs,   # list[str]
    ground_truth="...",                      # reference answer
)

print(result.faithfulness)
print(result.dangerous_downgrade)
```

<!-- TODO: replace with your real import path and function signature -->

This decoupling is intentional: the evaluator has no knowledge of how the
answer was produced, so it cannot be biased by the same retrieval mistakes
it is supposed to catch.

---

## 🔧 Design decisions

**Why LLM-as-a-Judge instead of embedding similarity?**
Cosine similarity cannot tell the difference between *"give paracetamol"* and
*"do not give paracetamol"* — the vectors are nearly identical. Clinical
correctness often lives in a single negation, so the evaluator has to
actually reason about the text.

**Why DeepSeek rather than the model under test?**
Using the same model as both author and judge produces inflated scores.
Keeping the judge on a different provider makes self-favoring bias much
harder.

**Why Pydantic on every judge response?**
LLM judges return malformed JSON often enough that it matters. Without schema
validation, a broken response silently becomes a `0` and quietly drags your
average down. Now it raises.

**Why temperature 0.0?**
An evaluation harness that returns different numbers on the same input is not
an evaluation harness.

<!-- TODO: add your own decisions here — e.g. the Implicit Entailment problem -->

---

## 📖 Engineering notes

Notes on the failure modes I hit while building this — including the
**Implicit Entailment** problem, where a judge marks a medically correct
answer as unfaithful because the supporting fact is implied by the context
rather than stated in it:

👉 [`docs/lessons_learned.md`](docs/lessons_learned.md)

---

## 🚧 Limitations

- Validated on one domain (pediatric triage) and one language pair. Scores on
  other clinical domains are unverified.
- Judge-model dependent. Changing the judge changes the absolute numbers;
  only compare runs that used the same judge and version.
- Evaluation set is small and hand-written, not clinician-reviewed.
- No inter-rater agreement study against human experts yet.

## 🗺️ Roadmap

- [ ] Publish to PyPI
- [ ] Human-vs-judge agreement study on the safety metric
- [ ] Support for additional judge providers
- [ ] Cost and latency reporting per run

---

## ⚠️ Disclaimer

Research and educational tool. Not a medical device, not validated for
clinical use, and not a substitute for expert human review of any system
that touches patient care.

## 📄 License

<!-- TODO: pick one — MIT is the usual default -->
