


# RAG_Eval: Clinical LLM Evaluation Framework 

A lightweight, independent evaluation framework designed to assess the quality, safety, and truthfulness of Medical RAG (Retrieval-Augmented Generation) systems. 

## 🎯 Why This Exists?
Building a RAG system in the healthcare domain is high-stakes. Evaluating it shouldn't be guesswork. `RAG_Eval` acts as an independent "inspector" using the LLM-as-a-Judge pattern (via DeepSeek API) to measure critical dimensions of a RAG pipeline without modifying the core application.

## 📊 Performance Dashboard (Example)
*This is an evaluation run on my pediatric triage system, ensuring 100% Clinical Safety and 0 dangerous downgrades:*

![Evaluation Dashboard](path_to_your_screenshot.png)



## ⚙️ Core Evaluation Metrics

1. **Faithfulness:** Does the generated answer strictly rely on the retrieved context, or did the LLM hallucinate?
2. **Answer Relevance:** Does the answer actually address the user's medical query without wandering off-topic?
3. **Context Precision:** Did the retriever fetch useful, highly relevant information, or noisy/irrelevant chunks?
4. **Context Recall:** Did the retriever successfully fetch *all* the necessary facts required to answer the query compared to the ground truth?

## 🚀 Quick Start

**1. Install Requirements:**
```bash
pip install openai pydantic python-dotenv
```

**2. Set up Environment:**
Create a `.env` file in the root directory:
```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

**3. Run the Evaluation:**
```bash
python test_all_metrics.py
```

## 📖 Deep Dives & Engineering Stories
I built this tool while developing MediMama (a pediatric RAG system). During the process, I encountered fascinating challenges with LLMs acting as judges in medical scenarios. 

Read about my engineering journey and the challenges of **"Implicit Entailment"** here:
👉 [Lessons Learned & Engineering Story](docs/lessons_learned.md)

## ⚠️ Disclaimer
This is an evaluation tool built for research and educational purposes. It is not intended for direct clinical decision-making.
