"""
analytic.py — Problem 13: Latency Budget of a Microservice Architecture
COE 562 Engineering Systems Design and Modelling, KNUST

Covers parts (a) and (b):
  (a) Open tandem M/M/1 network: per-stage utilisation and mean E2E latency.
  (b) Bottleneck identification and critical arrival rate.

System: λ → Gateway(1ms) → Auth(3ms) → Business(8ms) → Database(12ms) → reply
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── System parameters ─────────────────────────────────────────────────────────
STAGES            = ['Gateway', 'Auth', 'Business', 'Database']
SERVICE_TIMES_MS  = np.array([1.0, 3.0, 8.0, 12.0])   # mean service times in ms
LAMBDA_DEFAULT    = 200.0                               # given arrival rate (req/s)
P95_TARGET_MS     = 100.0    # assumed SLA: 95th-percentile end-to-end must be < 100 ms

# ── Basic M/M/1 formulas ──────────────────────────────────────────────────────

def service_rates():
    """Service rates μᵢ in req/s from mean service times in ms."""
    return 1000.0 / SERVICE_TIMES_MS


def utilisation(lam: float) -> np.ndarray:
    """Traffic intensity ρᵢ = λ/μᵢ at each stage."""
    return lam / service_rates()


def mm1_mean_sojourn_ms(lam: float, mu: float) -> float:
    """
    Mean sojourn time (queue + service) for a single M/M/1 stage.
    W = 1 / (μ(1 − ρ))   [seconds] → converted to ms.
    Returns np.inf if ρ ≥ 1 (unstable).
    """
    rho = lam / mu
    if rho >= 1.0:
        return np.inf
    return 1000.0 / (mu * (1.0 - rho))


def e2e_mean_ms(lam: float) -> float:
    """
    Mean end-to-end latency via Jackson's theorem:
        W_total = Σᵢ Wᵢ   (stages are independent M/M/1 queues).
    """
    mus = service_rates()
    return sum(mm1_mean_sojourn_ms(lam, mu) for mu in mus)


# ── Sojourn time distribution — hypoexponential ───────────────────────────────

def _hypoexponential_survival(t_ms: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Survival function P(W_total > t) for the sum of n independent
    exponentials Xᵢ ~ Exp(αᵢ) with distinct rates αᵢ (in ms⁻¹).

    Formula (partial-fraction expansion):
        P(W > t) = Σᵢ  Cᵢ · exp(−αᵢ t)
        Cᵢ = ∏_{j≠i} αⱼ / (αⱼ − αᵢ)
    """
    n = len(alphas)
    t = np.atleast_1d(np.asarray(t_ms, dtype=float))
    survival = np.zeros_like(t)
    for i in range(n):
        coeff = 1.0
        for j in range(n):
            if j == i:
                continue
            diff = alphas[j] - alphas[i]
            # Guard against near-equal rates (shouldn't occur with our data)
            if abs(diff) < 1e-12:
                diff = 1e-12
            coeff *= alphas[j] / diff
        survival += coeff * np.exp(-alphas[i] * t)
    return survival


def e2e_survival(t_ms: np.ndarray, lam: float) -> np.ndarray:
    """
    P(W_total > t) for end-to-end latency.
    Each stage sojourn ~ Exp(μᵢ(1−ρᵢ)), so the total is hypoexponential.
    """
    mus  = service_rates()
    rhos = lam / mus
    if np.any(rhos >= 1.0):
        return np.ones_like(np.atleast_1d(t_ms))
    # Effective rates in ms⁻¹
    alphas = mus * (1.0 - rhos) / 1000.0
    return _hypoexponential_survival(t_ms, alphas)


def e2e_cdf(t_ms: np.ndarray, lam: float) -> np.ndarray:
    """CDF of end-to-end latency F(t) = P(W_total ≤ t)."""
    return 1.0 - e2e_survival(t_ms, lam)


def e2e_p95_ms(lam: float) -> float:
    """
    95th-percentile end-to-end latency via numerical root-finding on the CDF.
    Returns np.inf if system is unstable.
    """
    mus  = service_rates()
    rhos = lam / mus
    if np.any(rhos >= 1.0):
        return np.inf
    try:
        t95 = brentq(lambda t: e2e_cdf(t, lam) - 0.95, 0.0, 1e5)
        return float(t95)
    except ValueError:
        return np.inf


# ── Bottleneck analysis ───────────────────────────────────────────────────────

def bottleneck() -> tuple[str, float]:
    """Stage with the lowest service rate — limits maximum stable throughput."""
    mus  = service_rates()
    idx  = int(np.argmin(mus))
    return STAGES[idx], mus[idx]


def critical_lambda_stability() -> float:
    """Maximum arrival rate before the bottleneck saturates (ρ < 1)."""
    _, mu_bottleneck = bottleneck()
    return mu_bottleneck   # req/s


def critical_lambda_for_target(target_ms: float = P95_TARGET_MS) -> float | None:
    """
    Arrival rate at which the 95th-percentile E2E latency first exceeds target_ms.
    Returns None if the target is already violated at very low load.
    """
    lam_max = critical_lambda_stability() - 0.1
    try:
        return brentq(lambda lam: e2e_p95_ms(lam) - target_ms, 1.0, lam_max)
    except ValueError:
        return None


# ── Reporting ────────────────────────────────────────────────────────────────

def print_stage_summary(lam: float = LAMBDA_DEFAULT) -> None:
    mus  = service_rates()
    rhos = utilisation(lam)
    print(f"\n{'='*62}")
    print(f"  Part (a): M/M/1 Stage Summary    λ = {lam:.0f} req/s")
    print(f"{'='*62}")
    print(f"  {'Stage':<12} {'μ (req/s)':>12} {'ρ':>8} {'W_mean (ms)':>14}  {'Stable':>7}")
    print(f"  {'-'*60}")
    for name, mu, rho, s in zip(STAGES, mus, rhos, SERVICE_TIMES_MS):
        W = mm1_mean_sojourn_ms(lam, mu)
        stable = "YES" if rho < 1.0 else "NO  ⚠"
        W_str = f"{W:12.2f}" if np.isfinite(W) else "       ∞ (unstable)"
        print(f"  {name:<12} {mu:>12.1f} {rho:>8.3f} {W_str}  {stable:>7}")
    print(f"  {'-'*60}")
    W_tot = e2e_mean_ms(lam)
    W_str = f"{W_tot:.2f} ms" if np.isfinite(W_tot) else "∞ (system unstable)"
    print(f"  {'E2E Mean':>34}: {W_str}")
    p95   = e2e_p95_ms(lam)
    p95_str = f"{p95:.2f} ms" if np.isfinite(p95) else "∞ (system unstable)"
    print(f"  {'E2E P95 (analytic)':>34}: {p95_str}")
    print(f"{'='*62}\n")


def print_bottleneck_analysis() -> None:
    bname, bmu = bottleneck()
    lam_stab   = critical_lambda_stability()
    lam_target = critical_lambda_for_target(P95_TARGET_MS)
    print(f"\n{'='*62}")
    print(f"  Part (b): Bottleneck Analysis")
    print(f"{'='*62}")
    print(f"  Bottleneck stage   : {bname}")
    print(f"  Bottleneck capacity: {bmu:.1f} req/s  (service time {1000/bmu:.0f} ms)")
    print(f"  Max stable λ       : {lam_stab:.1f} req/s")
    if lam_target:
        print(f"  P95 > {P95_TARGET_MS:.0f} ms when λ  > {lam_target:.1f} req/s")
    print(f"  At λ = {LAMBDA_DEFAULT:.0f} req/s: system is UNSTABLE (ρ_DB = {LAMBDA_DEFAULT/bmu:.2f})")
    print(f"{'='*62}\n")


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_utilisation(save_path: str = "figures/fig1_utilisation.png") -> None:
    lam_stable = critical_lambda_stability()
    lams = np.linspace(5, lam_stable * 0.999, 400)
    rhos = np.array([utilisation(l) for l in lams])   # shape (N, 4)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for i, (name, color) in enumerate(zip(STAGES, colors)):
        ax.plot(lams, rhos[:, i], label=name, color=color, lw=2)
    ax.axhline(1.0, color='k', ls='--', lw=1, label='ρ = 1 (saturation)')
    ax.axvline(lam_stable, color='grey', ls=':', lw=1.2)
    ax.text(lam_stable - 1, 0.05, f'λ_max = {lam_stable:.0f}',
            ha='right', fontsize=9, color='grey')
    ax.set_xlabel('Arrival rate λ (req/s)')
    ax.set_ylabel('Utilisation ρ')
    ax.set_title('Per-Stage Utilisation vs Arrival Rate')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_latency_vs_lambda(save_path: str = "figures/fig2_latency_vs_lambda.png",
                           target_ms: float = P95_TARGET_MS) -> None:
    lam_stable = critical_lambda_stability()
    lams = np.linspace(5, lam_stable * 0.98, 400)
    means = [e2e_mean_ms(l) for l in lams]
    p95s  = [e2e_p95_ms(l)  for l in lams]

    lam_target = critical_lambda_for_target(target_ms)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lams, means, lw=2, label='Mean E2E latency')
    ax.plot(lams, p95s,  lw=2, ls='--', label='95th-percentile latency')
    ax.axhline(target_ms, color='red', ls=':', lw=1.5,
               label=f'P95 target = {target_ms:.0f} ms')
    if lam_target:
        ax.axvline(lam_target, color='red', ls=':', lw=1.2)
        ax.text(lam_target + 0.5, target_ms + 5,
                f'λ* = {lam_target:.1f}', color='red', fontsize=9)
    ax.axvline(lam_stable, color='grey', ls=':', lw=1.2)
    ax.set_xlabel('Arrival rate λ (req/s)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Analytic End-to-End Latency vs Arrival Rate')
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(500, target_ms * 5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cdf_analytic(lam: float = 50.0,
                      save_path: str = "figures/fig3_cdf_analytic.png") -> None:
    t_ms = np.linspace(0, 200, 1000)
    cdf  = e2e_cdf(t_ms, lam)
    # Per-stage CDFs
    mus  = service_rates()
    rhos = utilisation(lam)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: stage CDFs
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for i, (name, mu, rho, color) in enumerate(zip(STAGES, mus, rhos, colors)):
        alpha_i = mu * (1 - rho) / 1000   # ms^-1
        stage_cdf = 1 - np.exp(-alpha_i * t_ms)
        axes[0].plot(t_ms, stage_cdf, label=name, color=color, lw=1.5)
    axes[0].set_xlabel('t (ms)')
    axes[0].set_ylabel('P(W ≤ t)')
    axes[0].set_title(f'Per-Stage Sojourn CDFs  (λ = {lam:.0f} req/s)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Right: total E2E CDF
    axes[1].plot(t_ms, cdf, lw=2, color='black', label='E2E (hypoexponential)')
    p95 = e2e_p95_ms(lam)
    axes[1].axvline(p95, color='red', ls='--', lw=1.5, label=f'P95 = {p95:.1f} ms')
    axes[1].axhline(0.95, color='red', ls=':', lw=1)
    axes[1].set_xlabel('t (ms)')
    axes[1].set_ylabel('P(W_total ≤ t)')
    axes[1].set_title(f'End-to-End Latency CDF  (λ = {lam:.0f} req/s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    print_stage_summary(lam=LAMBDA_DEFAULT)
    print_stage_summary(lam=50.0)
    print_bottleneck_analysis()

    print("Generating analytic figures...")
    plot_utilisation()
    plot_latency_vs_lambda()
    plot_cdf_analytic(lam=50.0)
    print("Done.\n")
