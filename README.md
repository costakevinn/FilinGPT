# 🚀 FilinGPT — Financial Language Model Built from Scratch

A byte-level autoregressive language model trained on real SEC 10-K filings, implemented entirely in Python using NumPy.

FilinGPT is not a framework wrapper — it is a full end-to-end machine learning system designed from data ingestion to inference, emphasizing statistical rigor, modular architecture, and reproducible experimentation.

Author: Kevin Mota da Costa

Portfolio: [https://costakevinn.github.io](https://costakevinn.github.io)

LinkedIn: [https://linkedin.com/in/costakevinnn](https://linkedin.com/in/costakevinnn/)

---

## 🎯 Project Purpose

FilinGPT was built to deeply explore how language models work at a systems level — without relying on high-level deep learning abstractions.

The project integrates:

* Financial document processing (SEC 10-K)
* Structured ETL and dataset layering
* Custom neural network implementation (NumPy)
* Controlled training experiments
* Quantitative evaluation (loss & perplexity)
* Reproducible ML workflows

It reflects my focus on building machine learning systems that combine statistical modeling with disciplined data engineering.

---

## 🧠 System Architecture

The system follows a structured ML lifecycle:

Raw SEC 10-K filings
→ Section extraction (10-K & MD&A)
→ Bronze / Silver / Gold dataset layering
→ Chunking & tokenization
→ Batch construction
→ Autoregressive model training
→ Evaluation & metric reporting
→ Inference / interactive generation

### Repository Structure

```
data/     → Layered financial datasets
etl/      → Document extraction & processing
prep/     → Chunking, tokenization, batching
core/     → Model definition, training loop, inference
reports/  → Training metrics & comparisons
app/      → Interactive chat interface
```

The architecture separates data engineering from modeling logic, enabling reproducibility and controlled experimentation.

---

## 🏗 Model Design

* Byte-level tokenization
* Custom autoregressive MLP architecture
* Manual training loop implementation
* Explicit loss tracking
* Controlled hyperparameter configuration

The objective was to understand optimization dynamics, convergence behavior, and financial language structure from first principles.

---

## 📊 Training Results

| Metric           | Baseline (200 steps) | Financial Model (100k steps) |
| ---------------- | -------------------- | ---------------------------- |
| Final Loss       | 5.3182               | 0.8071                       |
| Final Perplexity | 204.02               | **2.24**                     |

The trained model demonstrates significant convergence and learns structured financial terminology, formatting patterns, and document style.

---

## 📈 Training Dynamics

### Perplexity Progression

![Financial Perplexity](reports/financial_ppl_full.png)

### Loss Comparison (Log Scale)

![Loss Comparison](reports/compare_loss_logx.png)

The comparison highlights how training depth and dataset quality influence convergence stability and predictive confidence.

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

The baseline model produces incoherent sequences, while the trained financial model generates structured, domain-aligned financial text.

---

## 📚 Engineering Decisions

* Built with NumPy to expose internal model mechanics
* Structured dataset layering for traceable data evolution
* Explicit hyperparameter control for training stability
* Separation of data, model, and interface layers
* Dockerized environment for reproducibility

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
