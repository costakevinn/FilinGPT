# FilinGPT — Financial LLM & End-to-End ML System

Financial domain Language Model (LLM) trained on SEC 10-K filings, built from scratch in Python using NumPy.

FilinGPT demonstrates full ML system engineering: financial data ingestion, ETL processing, dataset construction, neural network training, evaluation, and text generation in a modular, reproducible architecture.

---

## What This Project Demonstrates

- End-to-end Machine Learning pipeline (raw 10-K → model inference)
- Financial NLP modeling
- Custom neural network training (NumPy, no high-level DL frameworks)
- Autoregressive LLM-style generation
- Structured data engineering (bronze → silver → gold layers)
- Evaluation using cross-entropy and perplexity
- Dockerized and reproducible experimentation
- Retrieval-compatible document preprocessing (RAG-ready architecture)

---

## System Architecture

Raw 10-K filings  
→ ETL extraction (10-K & MDA sections)  
→ Dataset construction & chunking  
→ Byte-level tokenization  
→ Batch generation  
→ Model training  
→ Evaluation & reporting  
→ Inference / text generation  

```

data/   → Financial document layers (bronze/silver/gold)
etl/    → 10-K extraction pipeline
prep/   → Chunking, tokenization, batching
core/   → Model, training loop, inference
reports/→ Metrics & comparisons
app/    → Interactive chat interface

````

Modular design separating data engineering, model training, and inference layers.

---

## Training Results

| Metric | Baseline (200 steps) | Financial Model (100k steps) |
|--------|----------------------|------------------------------|
| Final Loss | 5.3182 | 0.8071 |
| Final Perplexity | 204.02 | **2.24** |

Perplexity reduced from 258 → 2.24, demonstrating strong convergence and structured financial language acquisition.

---

## Training Dynamics

### Perplexity Progression

![Financial Perplexity](reports/financial_ppl_full.png)

### Loss Comparison (Log Scale)

![Loss Comparison](reports/compare_loss_logx.png)

---

## Generation Quality

<table>
<tr>
<td align="center"><b>Baseline (200 steps)</b></td>
<td align="center"><b>Financial Model (100k steps)</b></td>
</tr>
<tr>
<td><img src="reports/baseline_chat.png" width="100%"/></td>
<td><img src="reports/financial_chat.png" width="100%"/></td>
</tr>
</table>

The baseline model produces incoherent byte-level output, while the trained financial model generates structured financial language with recognizable terminology and formatting patterns.

---

## Key ML & Engineering Concepts

LLM Fundamentals • Autoregressive Modeling • Financial NLP • ETL Pipelines • Data Layering • Model Training • Optimization • Evaluation Metrics • Reproducible ML Systems • Dockerized Execution

---

## Tech Stack

Python • NumPy • Structured Data Pipelines • Docker • Custom Training Loop • Financial Document Processing

---

## Run with Docker

```bash
docker build -t filingpt .
docker run --rm -it -v "$(pwd):/app" -w /app filingpt python -m app.chat
````

---

## License

MIT License
