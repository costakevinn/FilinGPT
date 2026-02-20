## Baseline vs Financial

| Field | baseline_v1 | financial_v1 |
|---|---:|---:|
| steps | 200 | 100000 |
| lr | 0.0100 | 0.0200 |
| seed | 42 | 42 |
| batch_size | 32 | 32 |
| vocab_size | 258 | 258 |
| ctx_len | 16 | 16 |
| seq_len | 256 | 256 |
| n_sequences | 609 | 609 |
| start_loss | 5.5550 | 5.5550 |
| final_loss | 5.3182 | 0.8071 |
| start_ppl (exp(loss)) | 258.53 | 258.53 |
| final_ppl (exp(loss)) | 204.02 | 2.24 |
| ppl_reduction_pct | 21.08% | 99.13% |

### Plots

- `compare_loss_first_200.png`: baseline vs financial (first 200 steps)
- `compare_ppl_first_200.png`: perplexity baseline vs financial (first 200 steps)
- `financial_loss_full.png`: financial loss (full training)
- `financial_ppl_full.png`: financial perplexity (full training)
- `compare_loss_logx.png`: baseline vs financial with log-scaled x-axis
- `compare_ppl_logx.png`: perplexity baseline vs financial with log-scaled x-axis
