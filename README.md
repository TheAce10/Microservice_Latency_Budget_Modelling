# Microservice Latency Budget Modelling

**Queueing-theoretic analysis and discrete-event simulation of a four-stage microservice pipeline**

Bless Elikem Krapah | PG4346325 | MPhil Computer Engineering, KNUST
*COE 562 — Engineering Systems Design and Modelling, Final Modelling Assessment (Problem 13)*

---

## System

Four-stage open tandem queueing network modelled as independent M/M/1 queues (Jackson's theorem):

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

| File | Purpose |
|---|---|
| `analytic.py` | M/M/1 analytics: utilisation, mean sojourn, hypoexponential CDF, P95 |
| `simulation.py` | SimPy discrete-event simulation: 3 service distributions, 8 replications |
| `capacity.py` | Erlang-C, Strategy A (replicate) vs Strategy B (speed up) |
| `main.py` | Orchestrates all parts |
| `figures/` | All 8 output figures |
| `report/` | 17-page PDF report and LaTeX source |

## Figure index

| Figure | Description |
|---|---|
| fig1_utilisation | Per-stage ρ vs λ — bottleneck identification |
| fig2_latency_vs_lambda | Analytic mean and P95 vs λ, SLA crossover |
| fig3_cdf_analytic | Hypoexponential E2E CDF at λ = 50 req/s |
| fig4_sim_vs_analytic_cdf | Simulation vs analytic CDF for 3 service-time distributions |
| fig5_p95_vs_lambda | Analytic vs simulated P95 sweep with confidence intervals |
| fig6_littles_law | Little's Law L = λW validation per stage |
| fig7_capacity_strategies | Per-stage latency: Strategy A vs Strategy B |
| fig8_capacity_sim | Empirical CDFs of both strategies at λ = 200 req/s |

## Key results

| Metric | Value |
|---|---|
| Bottleneck stage | Database (μ = 83.3 req/s) |
| Max stable arrival rate | 83.3 req/s |
| Mean E2E latency at λ = 50 req/s | 47.92 ms |
| P95 latency at λ = 50 req/s | 112.18 ms |
| P95 exceeds 100 ms target at | λ ≈ 45.1 req/s |
| Strategy A (replicate) E2E mean at λ = 200 | 32.47 ms |
| Strategy B (speed up) E2E mean at λ = 200 | 21.96 ms — **recommended** |

## Random seeds

Analytic code is fully deterministic. DES base seed = 1000 (declared in `simulation.py` line 32).
