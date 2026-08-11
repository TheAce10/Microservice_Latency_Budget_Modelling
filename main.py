"""
main.py 
Problem 13: Latency Budget of a Microservice Architecture
COE 562 Engineering Systems Design and Modelling, KNUST

Run this file to reproduce every result and figure for the report.
All random seeds are declared; re-running produces identical output.

Usage:
    python main.py [--quick]   (quick skips multi-λ simulation sweep)
"""

import os
import sys
import time

os.makedirs('figures', exist_ok=True)

print("=" * 62)
print("  COE 562: Problem 13: Microservice Latency Budget")
print("  Bless Elikem Krapah | PG4346325")
print("=" * 62)


############# Part (a) & (b): Analytics 
print("\n[1/4] Running analytic model (parts a & b)...")
t0 = time.time()

from analytic import (
    print_stage_summary, print_bottleneck_analysis,
    plot_utilisation, plot_latency_vs_lambda, plot_cdf_analytic,
    LAMBDA_DEFAULT
)

print_stage_summary(lam=LAMBDA_DEFAULT)   # shows system is overloaded
print_stage_summary(lam=50.0)             # stable reference point
print_bottleneck_analysis()

plot_utilisation(save_path='figures/fig1_utilisation.png')
plot_latency_vs_lambda(save_path='figures/fig2_latency_vs_lambda.png')
plot_cdf_analytic(lam=50.0, save_path='figures/fig3_cdf_analytic.png')

print(f"  Done in {time.time()-t0:.1f}s")


############# Parts (c) & (d): Simulation 
print("\n[2/4] Running DES simulation (parts c & d)...")
t0 = time.time()

from simulation import (
    print_simulation_summary, print_littles_law,
    plot_sim_vs_analytic_cdf, plot_p95_vs_lambda, plot_littles_law
)

print_simulation_summary(lam=50.0)
print_littles_law(lam=50.0)

plot_sim_vs_analytic_cdf(lam=50.0, save_path='figures/fig4_sim_vs_analytic_cdf.png')
plot_littles_law(lam=50.0, save_path='figures/fig6_littles_law.png')

if '--quick' not in sys.argv:
    print("  Running multi-λ P95 sweep (8 reps x 7 λ values x 2 dists)...")
    plot_p95_vs_lambda(save_path='figures/fig5_p95_vs_lambda.png')
else:
    print("  Skipping multi-λ sweep (--quick mode).")

print(f"  Done in {time.time()-t0:.1f}s")


############# Part (e): Capacity strategies 
print("\n[3/4] Evaluating capacity strategies (part e)...")
t0 = time.time()

from capacity import (
    print_capacity_analysis, plot_strategy_comparison, plot_capacity_cdf
)

print_capacity_analysis(lam=LAMBDA_DEFAULT)
plot_strategy_comparison(save_path='figures/fig7_capacity_strategies.png')
plot_capacity_cdf(save_path='figures/fig8_capacity_sim.png')

print(f"  Done in {time.time()-t0:.1f}s")


############# Summary 
print("\n[4/4] Summary")
print("=" * 62)
print("  Figures generated:")
for f in sorted(os.listdir('figures')):
    size_kb = os.path.getsize(os.path.join('figures', f)) // 1024
    print(f"    figures/{f}  ({size_kb} KB)")

print()
print("  Key results:")
from analytic import e2e_mean_ms, e2e_p95_ms, critical_lambda_stability, critical_lambda_for_target
lam_stab   = critical_lambda_stability()
lam_target = critical_lambda_for_target(100.0)
print(f"    Bottleneck         : Database  (μ = {lam_stab:.1f} req/s, s = 12 ms)")
print(f"    Max stable λ       : {lam_stab:.1f} req/s")
if lam_target:
    print(f"    P95 > 100 ms when  : λ > {lam_target:.1f} req/s")
print(f"    At λ = 50 req/s    : mean = {e2e_mean_ms(50):.1f} ms, P95 = {e2e_p95_ms(50):.1f} ms")
from capacity import min_servers_per_stage, extra_cost
ns = min_servers_per_stage(LAMBDA_DEFAULT)
print(f"    Servers needed (Bus/DB)    : {ns[2]}× Business, {ns[3]}× Database")
print(f"    Total extra server units   : {extra_cost(LAMBDA_DEFAULT)}")
print("=" * 62)
print("\nAll done. To reproduce: python main.py")
print("Seeds: analytic (deterministic), simulation base seed = 1000")
