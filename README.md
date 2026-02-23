# 🚀 FilinGPT — Financial Language Model Built from Scratch

A byte-level autoregressive language model trained on real SEC 10-K filings, implemented entirely in Python using NumPy.

FilinGPT is a full end-to-end machine learning system — from financial document ingestion to inference — designed with modular architecture, structured data pipelines, and reproducible experimentation.

Author: Kevin Mota da Costa
Portfolio: [https://costakevinn.github.io](https://costakevinn.github.io)
LinkedIn: [https://linkedin.com/in/SEUUSER](https://linkedin.com/in/SEUUSER)

---

## 🎯 Project Purpose

FilinGPT was built to explore language model training at a systems level, without relying on high-level deep learning frameworks.

The project integrates:

* Financial document extraction (SEC 10-K)
* Structured ETL pipelines
* Dataset layering (Bronze / Silver / Gold)
* Byte-level tokenization
* Custom neural network training (NumPy-based)
* Quantitative evaluation (loss & perplexity)
* Dockerized reproducibility

It reflects my approach to combining machine learning modeling with disciplined data engineering.

---

## 🧠 System Architecture

The ML pipeline follows a structured lifecycle:

Raw SEC 10-K filings
→ Section extraction (10-K & MD&A)
→ Dataset layering
→ Chunking & tokenization
→ Batch generation
→ Autoregressive model training
→ Evaluation & metric tracking
→ Inference / interactive generation

---

### Repository Structure

```
data/     → Layered financial datasets (bronze/silver/gold)
etl/      → 10-K extraction pipeline
prep/     → Chunking, tokenization, batching
core/     → Model, training loop, inference
reports/  → Metrics & comparisons
app/      → Interactive chat interface
```

The architecture clearly separates data engineering from modeling logic, enabling traceability and reproducibility.

---

## 🏗 Model Design

* Byte-level tokenizer
* Custom autoregressive MLP architecture
* Manual training loop implementation
* Explicit loss tracking
* Controlled hyperparameter configuration

The goal was to understand optimization dynamics, convergence behavior, and structured financial language learning from first principles.

---

## 📊 Training Results

| Metric           | Baseline (200 steps) | Financial Model (100k steps) |
| ---------------- | -------------------- | ---------------------------- |
| Final Loss       | 5.3182               | 0.8071                       |
| Final Perplexity | 204.02               | **2.24**                     |

The trained model significantly improves over the baseline, learning structured financial terminology and formatting patterns from real corporate filings.

---

## 📈 Training Dynamics

### Perplexity Progression

![Financial Perplexity](reports/financial_ppl_full.png)

### Loss Comparison (Log Scale)

![Loss Comparison](reports/compare_loss_logx.png)

The comparison highlights the impact of training depth and structured dataset construction on convergence stability.

---

## 🧪 Generation Quality

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

The baseline model produces incoherent output, while the trained financial model generates structured financial language with recognizable terminology and formatting patterns.

---

## 📚 Engineering Decisions

* Implemented with NumPy to expose internal model mechanics
* Layered dataset design for controlled data evolution
* Explicit hyperparameter control for stability
* Separation of data, model, and interface layers
* Dockerized environment for reproducible execution

---

## 🛠 Tech Stack

Python
NumPy
Structured ETL pipelines
Financial NLP
Docker
Modular ML architecture

---

## ▶ Run with Docker

```bash
docker build -t filingpt .
docker run --rm -it -v "$(pwd):/app" -w /app filingpt python -m app.chat
```

---

## 🌐 Portfolio

This project is part of my Machine Learning portfolio:
👉 [https://costakevinn.github.io](https://costakevinn.github.io)

---

## License

MIT License — see `LICENSE` for details.
