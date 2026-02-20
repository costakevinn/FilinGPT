# FilinGPT

Byte-level financial language model built from scratch using NumPy.

FilinGPT is an end-to-end autoregressive language modeling system trained on SEC 10-K filings.  
The project demonstrates custom neural network implementation, financial NLP preprocessing, and quantitative evaluation without relying on deep learning frameworks.

---

## 🚀 Highlights

- Custom MLP-based autoregressive model (NumPy only)
- Byte-level tokenization (vocab size = 258)
- Context window: 16 tokens
- Cross-entropy training with perplexity tracking
- Full ETL pipeline (10-K → MDA → chunks → batches)
- Dockerized execution
- Automated training comparison reports

---

## 📊 Results

| Metric | Baseline (200 steps) | Financial (100k steps) |
|--------|----------------------|------------------------|
| Final Loss | 5.3182 | 0.8071 |
| Final Perplexity | 204.02 | **2.24** |
| Perplexity Reduction | 21% | **99.13%** |

The model reduced perplexity from 258 → 2.24, demonstrating strong convergence and structured financial language acquisition.

---

## 📈 Training Dynamics

### Perplexity (100k steps)

![Financial Perplexity](reports/financial_ppl_full.png)

### Loss Comparison (log scale)

![Loss Comparison](reports/compare_loss_logx.png)

---

## 💬 Generation Quality

### Baseline (200 steps)

![Baseline Output](reports/baseline_chat.png)

### Financial Model (100k steps)

![Financial Output](reports/financial_chat.png)

The financial checkpoint produces structured financial English with recognizable domain terminology.

---

## 🏗 System Architecture

```

data/      → Financial document pipeline (bronze/silver/gold)
etl/       → 10-K & MDA extraction
prep/      → Chunking, tokenization, batching
core/      → Model, training loop, inference
scripts/   → Automated training comparison reports

````

Designed for reproducibility and modular ML experimentation.

---

## 🧠 ML & Engineering Concepts Demonstrated

- Autoregressive language modeling
- Cross-entropy optimization
- Perplexity evaluation
- Byte-level NLP
- Data pipeline engineering
- Domain-specific language learning
- Reproducible ML systems
- RAG-ready financial document processing pipeline

The dataset structure and preprocessing stages make the system directly extensible to Retrieval-Augmented Generation (RAG) architectures.

---

## 🐳 Run with Docker

```bash
docker build -t filingpt .
docker run --rm -it -v "$(pwd):/app" -w /app filingpt python -m app.chat
````

---

## 📄 License

This project is licensed under the MIT License.

