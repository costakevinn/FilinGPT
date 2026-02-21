# FilinGPT — Financial Language Model & ML Pipeline

Financial domain language model trained on SEC 10-K filings, built from scratch in Python using NumPy.

FilinGPT demonstrates full end-to-end machine learning system design: financial data ingestion, structured preprocessing, dataset construction, neural network training, evaluation, and text generation in a modular and reproducible architecture.

---

## What This Project Demonstrates

- End-to-end Machine Learning pipeline (raw financial documents → trained model → inference)
- Financial NLP modeling using real SEC 10-K filings
- Custom neural network implementation (NumPy-based)
- Structured data engineering (document extraction, dataset construction, batching)
- Model training, evaluation, and performance comparison
- Modular architecture separating data, training, and inference layers
- Dockerized and reproducible experimentation

---

## System Architecture

Raw 10-K filings  
→ ETL extraction (10-K & MDA sections)  
→ Dataset construction & chunking  
→ Tokenization & batch generation  
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

Designed as a modular ML system combining data engineering and model development.

---

## Training Results

| Metric | Baseline (200 steps) | Financial Model (100k steps) |
|--------|----------------------|------------------------------|
| Final Loss | 5.3182 | 0.8071 |
| Final Perplexity | 204.02 | **2.24** |

The trained model significantly improves over the baseline, learning structured financial language patterns from real corporate filings.

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

The baseline model produces mostly incoherent output, while the trained financial model generates structured financial language with recognizable terminology and formatting patterns.

---

## Key Skills Demonstrated

**Machine Learning**
Neural Network Training • Language Modeling • Model Evaluation • Optimization • Predictive Systems

**Data Engineering**
ETL Pipeline Design • Dataset Construction • Data Layering (Bronze/Silver/Gold) • Structured Preprocessing • Reproducible ML Workflows

---

## Tech Stack

Python • NumPy • Structured ETL Pipelines • Modular ML Architecture • Docker

---

## Run with Docker

```bash
docker build -t filingpt .
docker run --rm -it -v "$(pwd):/app" -w /app filingpt python -m app.chat
````

---

## License

MIT License
