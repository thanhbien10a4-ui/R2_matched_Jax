"""
robust_r2_matched_cost_confirmatory/scripts/cost_power_table.py:
Computes exact noncentral-t marginal power, Frechet lower bound, and bivariate
Monte Carlo (B=60,000) joint Berger IUT power for R1..R8 plus 10-bps cost-stress R9
at N=24 and alpha=0.025 one-sided.
Reuses nct_power_lower_tail and t_ppf from robust_r2_matched_confirmatory/scripts/sample_size_and_power.py.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
COST_CONF_DIR = SCRIPTS_DIR.parent
EXP_DIR = COST_CONF_DIR.parent
R2_CONF_SCRIPTS = EXP_DIR / "robust_r2_matched_confirmatory" / "scripts"
if str(R2_CONF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(R2_CONF_SCRIPTS))

from sample_size_and_power import nct_power_lower_tail, t_ppf

# Pre-run sensitivity assumptions: R1..R8 preserve zero-cost planning assumptions;
# R9 reflects 10-bps fee uncertainty (shrunk H1b margin + inflated paired variance, never yet measured).
POWER_SCENARIOS = [
    ("R1_ZeroCost_PointEst", -0.0048, 0.0008, 0.00000, 0.000050, 0.0),
    ("R2_Realistic_Base", -0.0038, 0.0012, 0.00001, 0.000065, 0.0),
    ("R3_Conservative_Design", -0.0030, 0.0015, 0.00002, 0.000075, 0.0),
    ("R4_NegCorr_Conservative", -0.0030, 0.0015, 0.00002, 0.000075, -0.3),
    ("R5_PosCorr_Conservative", -0.0030, 0.0015, 0.00002, 0.000075, +0.4),
    ("R6_HighDrift_Stress", -0.0025, 0.0018, 0.00003, 0.000085, 0.0),
    ("R7_HeavyVariance_FIXEDlike", -0.0030, 0.0025, 0.00002, 0.000110, 0.0),
    ("R8_Worst_40pctMargin_FIXEDsd", -0.0020, 0.0025, 0.00004, 0.000120, -0.3),
    ("R9_Cost10_Unmeasured_FeeStress", -0.0020, 0.0025, 0.00006, 0.000150, -0.3),
]


def mc_joint_power(
    n: int,
    mu_a: float,
    sd_a: float,
    mu_b: float,
    sd_b: float,
    rho_ab: float,
    eps_b: float = 0.00010,
    alpha: float = 0.025,
    n_mc: int = 60000,
    seed: int = 20261120,
) -> float:
    rng = np.random.default_rng(seed)
    c_crit = -t_ppf(1.0 - alpha, n - 1)
    cov = np.array([[sd_a**2, rho_ab * sd_a * sd_b], [rho_ab * sd_a * sd_b, sd_b**2]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_mc, n, 2))
    x = z @ L.T
    da = x[:, :, 0] + mu_a
    db = x[:, :, 1] + mu_b
    ta = np.mean(da, axis=1) / (np.std(da, axis=1, ddof=1) / math.sqrt(n))
    tb = (np.mean(db, axis=1) - eps_b) / (np.std(db, axis=1, ddof=1) / math.sqrt(n))
    return float(np.mean((ta < c_crit) & (tb < c_crit)))


def required_n_h1b(margin_b: float, sd_b: float, target_power: float = 0.90, alpha: float = 0.025) -> int:
    for n in range(4, 500):
        if nct_power_lower_tail(n, margin_b, sd_b, alpha) >= target_power:
            return n
    return 500


def build_power_table(n: int = 24, eps_b: float = 0.00010, alpha: float = 0.025) -> Path:
    rows = []
    for name, mu_a, sd_a, mu_b, sd_b, rho in POWER_SCENARIOS:
        pa = nct_power_lower_tail(n, -mu_a, sd_a, alpha)
        pb = nct_power_lower_tail(n, eps_b - mu_b, sd_b, alpha)
        frechet = max(0.0, pa + pb - 1.0)
        mc_p = mc_joint_power(n, mu_a, sd_a, mu_b, sd_b, rho, eps_b=eps_b, alpha=alpha, seed=20261120)
        n90_b = required_n_h1b(eps_b - mu_b, sd_b, 0.90, alpha)
        rows.append(
            {
                "scenario": name,
                "n": n,
                "alpha_one_sided": alpha,
                "eps_b": eps_b,
                "mu_a": mu_a,
                "sd_a": sd_a,
                "mu_b": mu_b,
                "sd_b": sd_b,
                "rho_ab": rho,
                "margin_b": round(eps_b - mu_b, 8),
                "p_a_nct": round(pa, 5),
                "p_b_nct": round(pb, 5),
                "p_joint_frechet": round(frechet, 5),
                "p_joint_mc60k": round(mc_p, 5),
                "n_for_90pct_h1b": n90_b,
            }
        )
    out = COST_CONF_DIR / "SAMPLE_SIZE_SENSITIVITY_COST.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


if __name__ == "__main__":
    out_p = build_power_table()
    print("Saved cost power sensitivity table to", out_p)
    print(out_p.read_text(encoding="utf-8"))
