"""
simulation.py: Problem 13: Latency Budget of a Microservice Architecture
COE 562 Engineering Systems Design and Modelling, KNUST

Covers parts (c) and (d):
  (c) Discrete-event simulation with non-exponential service times;
      compare 95th-percentile vs analytic M/M/1 prediction.
  (d) Validate against Little's Law and against the analytic model.

Uses SimPy 4.x for discrete-event simulation.
Random seed is fixed for reproducibility; every seed is declared.
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as ss
from typing import Optional

from analytic import (
    STAGES, SERVICE_TIMES_MS, LAMBDA_DEFAULT,
    e2e_mean_ms, e2e_p95_ms, e2e_cdf, service_rates, utilisation
)

############# Simulation settings 
SIM_TIME_S  = 300.0    # total simulation clock (seconds)
WARM_UP_S   = 30.0     # discard first 30 s (transient phase)
N_REPS      = 8        # independent replications (different seeds)
BASE_SEED   = 1000     # seeds are BASE_SEED, BASE_SEED+1, ..., BASE_SEED+N_REPS-1


############# Service-time distributions 

def make_service_fn(mean_ms: float, dist: str = 'exponential', rng=None):
    """
    Return a zero-argument callable that draws a service time in ms.

    dist options:
      'exponential'   - memoryless, CV = 1 (M/M/1 assumption)
      'lognormal'     - right-skewed, CV = 2 (common in practice)
      'deterministic' - constant (M/D/1 limit; minimum variance)
    """
    if rng is None:
        rng = np.random.default_rng()

    if dist == 'exponential':
        return lambda: rng.exponential(mean_ms)

    elif dist == 'lognormal':
        cv = 2.0
        sigma2 = np.log(cv**2 + 1.0)          # variance of log(X)
        mu_log  = np.log(mean_ms) - sigma2 / 2  # mean of log(X)
        sigma   = np.sqrt(sigma2)
        return lambda: rng.lognormal(mu_log, sigma)

    elif dist == 'deterministic':
        return lambda: mean_ms

    else:
        raise ValueError(f"Unknown distribution '{dist}'. "
                         "Choose 'exponential', 'lognormal', or 'deterministic'.")


############# SimPy tandem queue 

class TandemQueue:
    """
    Open tandem M/G/1 (or M/G/n) network with four stages in series.

    Parameters
    ----------
    env          : simpy.Environment
    lambda_rps   : arrival rate (req/s)
    service_ms   : mean service times at each stage (ms)
    dist         : service-time distribution ('exponential'|'lognormal'|'deterministic')
    n_servers    : list of server counts per stage, default [1,1,1,1]
    warm_up_ms   : observations before this time are discarded
    rng          : numpy Generator for reproducibility
    """

    def __init__(self, env, lambda_rps, service_ms, dist='exponential',
                 n_servers=None, warm_up_ms=0.0, rng=None):
        self.env         = env
        self.lambda_rps  = lambda_rps
        self.warm_up_ms  = warm_up_ms
        self.n_stages    = len(service_ms)
        ns = n_servers or [1] * self.n_stages
        self.resources   = [simpy.Resource(env, capacity=c) for c in ns]
        if rng is None:
            rng = np.random.default_rng()
        self.service_fns = [make_service_fn(s, dist, rng) for s in service_ms]
        self._rng        = rng

        # Collected metrics (after warm-up)
        self.e2e_latencies  = []            # ms
        self.stage_sojourns = [[] for _ in range(self.n_stages)]

        # For Little's Law: running area under queue-length curve
        self._queue_arrival_count = np.zeros(self.n_stages, dtype=float)
        self._queue_depart_count  = np.zeros(self.n_stages, dtype=float)

    ############# Request lifecycle 

    def _request_proc(self):
        t_arrive_system = self.env.now
        record = t_arrive_system >= self.warm_up_ms

        for i, (res, svc_fn) in enumerate(zip(self.resources, self.service_fns)):
            t_stage_arrive = self.env.now
            with res.request() as req:
                yield req
                svc = svc_fn()
                yield self.env.timeout(svc)
            t_stage_depart = self.env.now

            if record:
                self.stage_sojourns[i].append(t_stage_depart - t_stage_arrive)

        if record:
            self.e2e_latencies.append(self.env.now - t_arrive_system)

    ############# Poisson arrival process 

    def _arrival_proc(self, sim_end_ms):
        mean_iat_ms = 1000.0 / self.lambda_rps
        while self.env.now < sim_end_ms:
            yield self.env.timeout(self._rng.exponential(mean_iat_ms))
            self.env.process(self._request_proc())

    def run(self, sim_time_s: float = SIM_TIME_S):
        sim_end_ms = sim_time_s * 1000.0
        self.env.process(self._arrival_proc(sim_end_ms))
        self.env.run(until=sim_end_ms)

    ############# Output statistics 

    def p95_ms(self) -> float:
        if len(self.e2e_latencies) < 20:
            return np.nan
        return float(np.percentile(self.e2e_latencies, 95))

    def mean_ms(self) -> float:
        if not self.e2e_latencies:
            return np.nan
        return float(np.mean(self.e2e_latencies))

    def empirical_cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (sorted_latencies, CDF_values) for the E2E distribution."""
        x = np.sort(self.e2e_latencies)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    def little_law_table(self) -> list[dict]:
        """
        Check L = λ·W at each stage.
        λ is approximated by count / effective_sim_time.
        L is approximated by λ·W.
        Returns a list of dicts with per-stage results.
        """
        rows = []
        effective_time_s = (SIM_TIME_S - WARM_UP_S)
        for i, name in enumerate(STAGES):
            soj = self.stage_sojourns[i]
            if len(soj) < 10:
                continue
            n_obs = len(soj)
            lam_meas  = n_obs / effective_time_s          # req/s
            W_ms      = float(np.mean(soj))               # ms
            L_littles = lam_meas * W_ms / 1000.0          # dimensionless
            rows.append({
                'Stage':     name,
                'n_obs':     n_obs,
                'λ_meas':    round(lam_meas, 2),
                'W_mean_ms': round(W_ms, 3),
                'L_littles': round(L_littles, 3),
            })
        return rows


############# Replication runner 

def run_one(lambda_rps: float, dist: str = 'exponential',
            n_servers: Optional[list] = None, seed: int = 0) -> TandemQueue:
    """Run a single replication and return the finished simulator object."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    sim = TandemQueue(env, lambda_rps, SERVICE_TIMES_MS, dist=dist,
                      n_servers=n_servers, warm_up_ms=WARM_UP_S * 1000, rng=rng)
    sim.run(SIM_TIME_S)
    return sim


def replicated_stats(lambda_rps: float, dist: str = 'exponential',
                     n_servers: Optional[list] = None,
                     n_reps: int = N_REPS) -> dict:
    """
    Run n_reps independent replications and return mean P95, mean E2E,
    and 95% confidence intervals.
    """
    p95s  = []
    means = []
    for r in range(n_reps):
        sim = run_one(lambda_rps, dist, n_servers, seed=BASE_SEED + r)
        p95s.append(sim.p95_ms())
        means.append(sim.mean_ms())

    p95s  = np.array(p95s)
    means = np.array(means)

    def ci95(arr):
        m  = np.mean(arr)
        se = ss.sem(arr)
        lo, hi = ss.t.interval(0.95, len(arr) - 1, loc=m, scale=se)
        return float(m), float(lo), float(hi)

    m_p95,  lo_p95,  hi_p95  = ci95(p95s)
    m_mean, lo_mean, hi_mean = ci95(means)

    return {
        'p95_mean': m_p95, 'p95_lo': lo_p95, 'p95_hi': hi_p95, 'p95_all': p95s,
        'mean_mean': m_mean, 'mean_lo': lo_mean, 'mean_hi': hi_mean,
    }


############# Printing 

def print_simulation_summary(lam: float = 50.0) -> None:
    print(f"\n{'='*62}")
    print(f"  Part (c): Simulation vs Analytic    λ = {lam:.0f} req/s")
    print(f"{'='*62}")

    analytic_mean = e2e_mean_ms(lam)
    analytic_p95  = e2e_p95_ms(lam)
    print(f"  Analytic mean E2E   : {analytic_mean:.2f} ms")
    print(f"  Analytic P95        : {analytic_p95:.2f} ms")
    print()

    for dist in ('exponential', 'lognormal', 'deterministic'):
        st = replicated_stats(lam, dist=dist)
        print(f"  Simulation ({dist:<13s})  "
              f"mean={st['mean_mean']:6.1f} ms   "
              f"P95={st['p95_mean']:6.1f} ms  "
              f"[{st['p95_lo']:.1f}, {st['p95_hi']:.1f}]")
    print(f"{'='*62}\n")


def print_littles_law(lam: float = 50.0) -> None:
    print(f"\n{'='*62}")
    print(f"  Part (d): Little's Law Check    λ = {lam:.0f} req/s")
    print(f"  L = λ·W  must hold at every stage")
    print(f"{'='*62}")
    sim = run_one(lam, dist='exponential', seed=BASE_SEED)
    rows = sim.little_law_table()
    mus  = service_rates()
    rhos = utilisation(lam)

    print(f"  {'Stage':<12} {'λ_meas':>10} {'W (ms)':>10} {'L=λW':>10}  {'Analytic W':>12}  {'Analytic L':>12}")
    print(f"  {'-'*68}")
    for i, row in enumerate(rows):
        rho   = rhos[i]
        W_ana = 1000.0 / (mus[i] * (1 - rho))
        L_ana = rho / (1 - rho)
        print(f"  {row['Stage']:<12} {row['λ_meas']:>10.1f} {row['W_mean_ms']:>10.2f} "
              f"{row['L_littles']:>10.3f}  {W_ana:>12.2f}  {L_ana:>12.3f}")
    print(f"{'='*62}\n")


############# Figures

def plot_sim_vs_analytic_cdf(lam: float = 50.0,
                              save_path: str = "figures/fig4_sim_vs_analytic_cdf.png") -> None:
    """Part (c): Compare empirical CDF (simulation) vs analytic (hypoexponential)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Analytic CDF
    t_ms = np.linspace(0, 300, 1000)
    ana_cdf = e2e_cdf(t_ms, lam)
    ax.plot(t_ms, ana_cdf, 'k-', lw=2, label='Analytic (M/M/1, hypoexponential)')

    # Simulation CDFs per distribution
    colors = {'exponential': '#1f77b4', 'lognormal': '#ff7f0e', 'deterministic': '#2ca02c'}
    labels = {'exponential': 'Sim: Exponential (CV=1)',
              'lognormal':   'Sim: Lognormal (CV=2)',
              'deterministic': 'Sim: Deterministic (CV=0)'}
    for dist, color in colors.items():
        sim = run_one(lam, dist=dist, seed=BASE_SEED)
        x, y = sim.empirical_cdf()
        ax.step(x, y, where='post', color=color, lw=1.5, alpha=0.8, label=labels[dist])

    p95_ana = e2e_p95_ms(lam)
    ax.axvline(p95_ana, color='red', ls='--', lw=1.2, label=f'Analytic P95 = {p95_ana:.1f} ms')
    ax.axhline(0.95, color='red', ls=':', lw=1)

    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title(f'Simulation vs Analytic CDF   (λ = {lam:.0f} req/s)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 300)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_p95_vs_lambda(save_path: str = "figures/fig5_p95_vs_lambda.png") -> None:
    """
    Compare analytic P95 and simulated P95 across a range of λ values.
    Shows how the simulation diverges from M/M/1 for heavier-tailed service times.
    """
    lam_stable = 83.0   # just below database saturation
    lams = [20, 30, 40, 50, 60, 70, 75]

    ana_p95 = [e2e_p95_ms(l) for l in lams]

    # Simulation P95 for exponential
    sim_exp  = []
    sim_lgn  = []
    sim_exp_lo, sim_exp_hi = [], []
    sim_lgn_lo, sim_lgn_hi = [], []

    for l in lams:
        st = replicated_stats(l, dist='exponential', n_reps=5)
        sim_exp.append(st['p95_mean']); sim_exp_lo.append(st['p95_lo']); sim_exp_hi.append(st['p95_hi'])
        st = replicated_stats(l, dist='lognormal', n_reps=5)
        sim_lgn.append(st['p95_mean']); sim_lgn_lo.append(st['p95_lo']); sim_lgn_hi.append(st['p95_hi'])

    lams     = np.array(lams)
    sim_exp  = np.array(sim_exp)
    sim_lgn  = np.array(sim_lgn)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lams, ana_p95,  'k-',  lw=2, label='Analytic P95 (M/M/1)')
    ax.plot(lams, sim_exp,  'b-o', lw=1.5, ms=5, label='Sim: Exponential (CV=1)')
    ax.fill_between(lams, sim_exp_lo, sim_exp_hi, alpha=0.15, color='blue')
    ax.plot(lams, sim_lgn,  'r--s', lw=1.5, ms=5, label='Sim: Lognormal (CV=2)')
    ax.fill_between(lams, sim_lgn_lo, sim_lgn_hi, alpha=0.15, color='red')

    ax.axhline(100.0, color='grey', ls=':', lw=1.2, label='SLA target = 100 ms')
    ax.set_xlabel('Arrival rate λ (req/s)')
    ax.set_ylabel('95th-percentile latency (ms)')
    ax.set_title('Analytic vs Simulated P95 Latency')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_littles_law(lam: float = 50.0,
                     save_path: str = "figures/fig6_littles_law.png") -> None:
    """Part (d): Bar chart of L_measured vs L = λW (analytic) per stage."""
    mus  = service_rates()
    rhos = utilisation(lam)
    L_ana = rhos / (1.0 - rhos)

    sim   = run_one(lam, dist='exponential', seed=BASE_SEED)
    rows  = sim.little_law_table()
    L_sim = np.array([r['L_littles'] for r in rows])

    x = np.arange(len(STAGES))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, L_ana, w, label="Analytic  L = ρ/(1-ρ)", color='#1f77b4')
    ax.bar(x + w/2, L_sim, w, label="Sim  L = λ·W (Little's Law)", color='#ff7f0e', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel('Mean number in system L')
    ax.set_title(f"Little's Law Validation   (λ = {lam:.0f} req/s, M/M/1 exponential)")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    print_simulation_summary(lam=50.0)
    print_littles_law(lam=50.0)

    print("Generating simulation figures (this may take a minute)...")
    plot_sim_vs_analytic_cdf(lam=50.0)
    plot_p95_vs_lambda()
    plot_littles_law(lam=50.0)
    print("Done.\n")
