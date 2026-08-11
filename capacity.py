"""
capacity.py: Problem 13: Latency Budget of a Microservice Architecture
COE 562 Engineering Systems Design and Modelling, KNUST

Covers part (e):
  Compare two equal-cost capacity strategies to handle λ = 200 req/s.

  At λ = 200 req/s, two stages are overloaded:
    Business  μ = 125 req/s  ρ = 1.60  (needs ≥ 2 servers)
    Database  μ = 83.3 req/s ρ = 2.40  (needs ≥ 3 servers)

  Strategy A: Replicate:
    Add the minimum number of servers at each overloaded stage (M/M/N).
    Business → M/M/2,  Database → M/M/3.
    Total extra cost = 1 (bus) + 2 (db) = 3 extra server units.

  Strategy B: Speed up:
    Replace each overloaded stage with ONE faster server at equal cost.
    Spend the same 3 server-units as speedup factor:
    Business → 2× faster (s = 4 ms),  Database → 3× faster (s = 4 ms).
    Both stay single-server M/M/1 with no coordination overhead.

  The primary bottleneck is the Database (lowest capacity 83.3 req/s),
  so the analysis focuses there while fixing Business as a prerequisite.
"""

import numpy as np
import matplotlib.pyplot as plt
import simpy
from math import ceil, factorial

from analytic import (
    STAGES, SERVICE_TIMES_MS, LAMBDA_DEFAULT,
    service_rates, mm1_mean_sojourn_ms
)
from simulation import TandemQueue, BASE_SEED, SIM_TIME_S, WARM_UP_S

# ── M/M/N: Erlang-C and sojourn time ─────────────────────────────────────────

def erlang_c(n: int, a: float) -> float:
    """P(arriving customer must wait) for M/M/n with offered load a = λ/μ."""
    rho = a / n
    if rho >= 1.0:
        raise ValueError(f"Unstable: a={a:.2f}, n={n}, ρ={rho:.3f}")
    sum_k   = sum(a**k / factorial(k) for k in range(n))
    last    = (a**n / factorial(n)) * (n / (n - a))
    return last / (sum_k + last)


def mmn_mean_sojourn_ms(lam: float, mu_rps: float, n: int) -> float:
    """
    Mean sojourn time (ms) for M/M/n.
    W = C(n,a)/(n·μ·(1−ρ)) + 1/μ
    """
    a   = lam / mu_rps
    rho = a / n
    if rho >= 1.0:
        return np.inf
    C    = erlang_c(n, a)
    Wq_s = C / (n * mu_rps * (1.0 - rho))
    return (Wq_s + 1.0 / mu_rps) * 1000.0   # ms


# ── Minimum servers to stabilise each stage ───────────────────────────────────

def min_servers_per_stage(lam: float) -> list[int]:
    """
    Minimum integer N at each stage so that N servers handle lam
    with per-server utilisation ρ < 1.
    For stable stages, N = 1.
    """
    mus = service_rates()
    ns  = []
    for mu in mus:
        rho = lam / mu
        if rho < 1.0:
            ns.append(1)
        else:
            ns.append(int(ceil(lam / mu)) + 1)
    return ns


def extra_cost(lam: float) -> int:
    """Total extra server units beyond single-server baseline."""
    return sum(max(0, n - 1) for n in min_servers_per_stage(lam))


# ── Strategy A — Replicate all overloaded stages (M/M/N) ─────────────────────

def strategy_A_e2e_mean(lam: float) -> tuple[float, list[int]]:
    """
    Add minimum servers at every overloaded stage; all others keep 1 server.
    Returns (mean E2E latency in ms, list of server counts).
    """
    ns  = min_servers_per_stage(lam)
    mus = service_rates()
    total = 0.0
    for i, (mu, n) in enumerate(zip(mus, ns)):
        total += mmn_mean_sojourn_ms(lam, mu, n)
    return total, ns


# ── Strategy B — Speed up all overloaded stages (single fast server) ──────────

def strategy_B_e2e_mean(lam: float) -> tuple[float, list[float]]:
    """
    At each overloaded stage, replace the stage with ONE server that is
    N× faster, where N is the minimum replica count for Strategy A.
    Total cost (server units) is identical to Strategy A.
    Returns (mean E2E latency in ms, list of speedup factors).
    """
    ns      = min_servers_per_stage(lam)
    mus     = service_rates()
    speedups = [max(1, n) for n in ns]            # equal cost units
    mus_new  = mus * np.array(speedups, dtype=float)
    total = 0.0
    for mu_new in mus_new:
        rho = lam / mu_new
        if rho >= 1.0:
            return np.inf, speedups
        total += mm1_mean_sojourn_ms(lam, mu_new)
    return total, speedups


# ── Reporting ────────────────────────────────────────────────────────────────

def print_capacity_analysis(lam: float = LAMBDA_DEFAULT) -> None:
    mus      = service_rates()
    rhos_raw = lam / mus
    ns_A     = min_servers_per_stage(lam)
    W_A, _   = strategy_A_e2e_mean(lam)
    W_B, speedups = strategy_B_e2e_mean(lam)

    print(f"\n{'='*68}")
    print(f"  Part (e): Capacity Strategy Comparison   λ = {lam:.0f} req/s")
    print(f"{'='*68}")

    print(f"\n  Baseline utilisation (single-server, λ = {lam:.0f} req/s):")
    print(f"  {'Stage':<12} {'μ (req/s)':>10} {'ρ':>8}  {'Status'}")
    print(f"  {'-'*50}")
    for name, mu, rho in zip(STAGES, mus, rhos_raw):
        status = "OK" if rho < 1.0 else "OVERLOADED ⚠"
        print(f"  {name:<12} {mu:>10.1f} {rho:>8.3f}  {status}")

    cost = extra_cost(lam)
    print(f"\n  Extra server units needed to stabilise: {cost}")

    print(f"\n  Strategy A: Replicate overloaded stages (M/M/N):")
    print(f"  {'Stage':<12} {'N servers':>10} {'ρ/server':>10} {'W_stage (ms)':>14}")
    print(f"  {'-'*52}")
    for name, mu, n in zip(STAGES, mus, ns_A):
        rho_per = (lam / mu) / n
        W_s = mmn_mean_sojourn_ms(lam, mu, n)
        print(f"  {name:<12} {n:>10d} {rho_per:>10.3f} {W_s:>14.2f}")
    print(f"  {'E2E mean':>36}: {W_A:.2f} ms")

    print(f"\n  Strategy B: Speed up overloaded stages (xN, single server):")
    print(f"  {'Stage':<12} {'Speedup':>10} {'New μ':>10} {'ρ':>8} {'W_stage (ms)':>14}")
    print(f"  {'-'*60}")
    mus_B = mus * np.array(speedups, dtype=float)
    for name, mu_b, spd in zip(STAGES, mus_B, speedups):
        rho_b = lam / mu_b
        W_b   = mm1_mean_sojourn_ms(lam, mu_b)
        print(f"  {name:<12} {spd:>10}× {mu_b:>10.1f} {rho_b:>8.3f} {W_b:>14.2f}")
    print(f"  {'E2E mean':>42}: {W_B:.2f} ms")

    print(f"\n  Equal cost (both use {cost} extra server units):")
    winner = "B (Speed up)" if W_B < W_A else "A (Replicate)"
    diff   = abs(W_A - W_B)
    print(f"  Strategy A mean E2E: {W_A:.2f} ms")
    print(f"  Strategy B mean E2E: {W_B:.2f} ms  (difference: {diff:.2f} ms)")
    print(f"\n  RECOMMENDATION: Strategy {winner}")
    print(f"  Reasoning:")
    if W_B < W_A:
        print(f"    - Strategy B achieves lower mean E2E latency.")
        print(f"    - A single fast server (M/M/1) has lower queuing variance")
        print(f"      than N slower servers (M/M/N) under the same offered load.")
        print(f"    - No need for load balancing logic across replicas.")
        print(f"    - Tail latency (P95, P99) is also better for M/M/1 fast.")
    else:
        print(f"    - Strategy A achieves lower mean E2E latency.")
        print(f"    - Replication provides fault tolerance (no single point of failure).")
        print(f"    - Multiple replicas reduce blast radius of individual failures.")
    print(f"{'='*68}\n")


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_strategy_comparison(lam: float = LAMBDA_DEFAULT,
                              save_path: str = "figures/fig7_capacity_strategies.png") -> None:
    """Bar chart comparing per-stage and total E2E latency for both strategies."""
    ns_A     = min_servers_per_stage(lam)
    mus      = service_rates()
    speedups = [max(1, n) for n in ns_A]
    mus_B    = mus * np.array(speedups, dtype=float)

    W_A_per = [mmn_mean_sojourn_ms(lam, mu, n) for mu, n in zip(mus, ns_A)]
    W_B_per = [mm1_mean_sojourn_ms(lam, mu_b) for mu_b in mus_B]

    x   = np.arange(len(STAGES) + 1)
    w   = 0.35
    labels = STAGES + ['TOTAL']
    A_bars = W_A_per + [sum(W_A_per)]
    B_bars = W_B_per + [sum(W_B_per)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_A = ax.bar(x - w/2, A_bars, w, label='Strategy A: Replicate (M/M/N)',
                    color='#1f77b4', edgecolor='white')
    bars_B = ax.bar(x + w/2, B_bars, w, label='Strategy B: Speed up (M/M/1 fast)',
                    color='#ff7f0e', edgecolor='white', alpha=0.9)

    for bar in bars_A:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars_B:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8)

    ax.axvline(len(STAGES) - 0.5, color='grey', ls=':', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean latency (ms)')
    ax.set_title(f'Capacity Strategy Comparison: Per-Stage and Total E2E   (λ = {lam:.0f} req/s)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_capacity_cdf(lam: float = LAMBDA_DEFAULT,
                      save_path: str = "figures/fig8_capacity_sim.png") -> None:
    """Simulate both strategies and compare empirical latency CDFs."""
    ns_A    = min_servers_per_stage(lam)
    mus     = service_rates()
    speedups = [max(1, n) for n in ns_A]
    svc_B   = SERVICE_TIMES_MS / np.array(speedups, dtype=float)

    # Strategy A simulation
    rng_A = np.random.default_rng(BASE_SEED)
    env_A = simpy.Environment()
    sim_A = TandemQueue(env_A, lam, SERVICE_TIMES_MS,
                        n_servers=ns_A, warm_up_ms=WARM_UP_S*1000, rng=rng_A)
    sim_A.run(SIM_TIME_S)

    # Strategy B simulation
    rng_B = np.random.default_rng(BASE_SEED + 100)
    env_B = simpy.Environment()
    sim_B = TandemQueue(env_B, lam, list(svc_B),
                        warm_up_ms=WARM_UP_S*1000, rng=rng_B)
    sim_B.run(SIM_TIME_S)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {
        'A': f'Strategy A: Replicate ({ns_A[1]}× Bus, {ns_A[3]}× DB)',
        'B': (f'Strategy B: Speed up '
              f'(Bus x{speedups[1]}, DB x{speedups[3]})')
    }
    for sim, key, color, ls in [(sim_A,'A','#1f77b4','-'),(sim_B,'B','#ff7f0e','--')]:
        x, y = sim.empirical_cdf()
        ax.step(x, y, where='post', lw=2, color=color, ls=ls, label=labels[key])
        p95 = sim.p95_ms()
        ax.axvline(p95, color=color, ls=':', lw=1.2)
        ax.text(p95 + 1, 0.45 + (0.08 if key=='A' else -0.08),
                f'P95 = {p95:.0f} ms', color=color, fontsize=9)

    ax.axhline(0.95, color='grey', ls=':', lw=1)
    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title(f'Simulated Latency CDFs: Both Strategies   (λ = {lam:.0f} req/s)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    print_capacity_analysis(lam=LAMBDA_DEFAULT)
    print("Generating capacity figures...")
    plot_strategy_comparison()
    plot_capacity_cdf()
    print("Done.\n")
