# Problem 13 — Latency Budget of a Microservice Architecture
**COE 562 Engineering Systems Design and Modelling, KNUST**
Student: Bless Elikem Krapah | PG4346325

## System
Four-stage open tandem queueing network:

```
λ=200 req/s → [Gateway 1ms] → [Auth 3ms] → [Business 8ms] → [Database 12ms] → reply
```

## Reproduce all results

```bash
pip install -r requirements.txt
python main.py          # full run (~5 minutes)
python main.py --quick  # skip multi-lambda sweep (~1 minute)
```

All random seeds are fixed. Every figure is deterministic.

## File layout

| File | Purpose | Report section |
|---|---|---|
| `analytic.py` | M/M/1 Jackson network analytics | Parts a, b |
| `simulation.py` | SimPy discrete-event simulation | Parts c, d |
| `capacity.py` | Capacity strategy comparison | Part e |
| `main.py` | Orchestrates all parts | — |
| `figures/` | All output figures | — |

## Figure index

| Figure | Description |
|---|---|
| fig1_utilisation | Per-stage ρ vs λ — shows bottleneck |
| fig2_latency_vs_lambda | Analytic mean and P95 vs λ |
| fig3_cdf_analytic | Hypoexponential E2E CDF at λ=50 |
| fig4_sim_vs_analytic_cdf | Sim vs analytic CDF for 3 service distributions |
| fig5_p95_vs_lambda | Analytic vs simulated P95 across λ range |
| fig6_littles_law | Little's Law L=λW bar chart per stage |
| fig7_capacity_strategies | E2E mean vs cost unit N for both strategies |
| fig8_capacity_sim | Empirical CDFs of both strategies at λ=200 |

## Key results (λ = 50 req/s stable reference)

- Bottleneck: **Database** (μ = 83.3 req/s)
- Max stable arrival rate: **83.3 req/s** (system overloaded at given λ = 200)
- P95 exceeds 100 ms target at **≈ 61 req/s**
- Minimum DB replicas to handle λ = 200: **3 servers**

## Declaration
AI assistance (Claude) used for code structure and equation derivations.
All model choices, assumptions, and results are understood and can be defended at viva.
Random seeds: analytic code is deterministic; DES base seed = 1000.
