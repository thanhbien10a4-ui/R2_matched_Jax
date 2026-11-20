"""
robust_r2_matched_confirmatory/scripts/sample_size_and_power.py

Reads development outputs (read-only, strictly stratified by stage and LR schedule)
and computes:
1. Stratified empirical summary of Delta = R2_matched_128 - A_ref_128 across dev stages.
2. Within-seed path-count sensitivity (5,000 vs 20,000 test paths from DIAG_PART1).
3. Marginal & simultaneous Intersection-Union (IUT) sample-size and power sensitivity table
   exported to `SAMPLE_SIZE_SENSITIVITY.csv`.

Strictly NO confirmatory test data or sealed HOLD_EVAL is accessed or simulated.
"""

import csv
import json
import math
from pathlib import Path
from typing import List, Dict
import numpy as np

CONF_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = CONF_DIR.parent


def load_csv(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- Exact Student-t & Noncentral-t Power (no external scipy dependency) ---
def _betacf(a: float, b: float, x: float) -> float:
    MAXIT = 250
    EPS = 3.0e-14
    FPMIN = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = FPMIN if abs(d) < FPMIN else 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else 1.0 / d
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else 1.0 / d
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        del_v = d * c
        h *= del_v
        if abs(del_v - 1.0) < EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: int) -> float:
    ib = _betainc(0.5 * df, 0.5, df / (df + x * x))
    return (1.0 - 0.5 * ib) if x >= 0.0 else (0.5 * ib)


def t_ppf(p: float, df: int) -> float:
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 200.0
    for _ in range(85):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nct_power_lower_tail(n: int, margin: float, sd: float, alpha: float) -> float:
    """
    Exact power of one-sided lower-tail paired t-test (H0: mu >= bound vs H1: mu < bound)
    when `bound - mu_true = margin` and paired difference SD is `sd`.
    """
    df = n - 1
    c_crit = -t_ppf(1.0 - alpha, df)
    ncp = -math.sqrt(n) * (margin / sd)
    k = 0.5 * df
    lgk = math.lgamma(k)
    m = 600
    a, b = 1e-6, max(4.5, 1.0 + 12.0 / math.sqrt(df))
    h = (b - a) / m
    total = 0.0
    for j in range(m + 1):
        y = a + j * h
        s = df * y * y
        log_f = -k * math.log(2.0) - lgk + (k - 1.0) * math.log(s) - 0.5 * s + math.log(2.0 * df * y)
        pdf_y = 0.0 if log_f < -700 else math.exp(log_f)
        phi = 0.5 * math.erfc(-(c_crit * y - ncp) / math.sqrt(2.0))
        w = 1.0 if (j == 0 or j == m) else (4.0 if j % 2 == 1 else 2.0)
        total += w * pdf_y * phi
    return float((h / 3.0) * total)


def simulate_joint_iut_power(
    n: int,
    mu_a: float,
    sd_a: float,
    mu_b: float,
    sd_b: float,
    eps_b: float,
    rho_ab: float,
    alpha_a: float,
    alpha_b: float,
    n_mc: int = 80000,
    sim_seed: int = 20261119,
) -> Dict[str, float]:
    """
    Monte Carlo joint power of the simultaneous Intersection-Union Test (H1_A AND H1_B)
    under assumed bivariate normal (DeltaHE_030, DeltaOCE_020) with correlation `rho_ab`.
    Strictly uses assumed parameters (zero confirmatory data touched).
    """
    rng = np.random.default_rng(sim_seed + n * 101 + int(round((rho_ab + 1.0) * 1000)))
    cov = np.array([[sd_a * sd_a, rho_ab * sd_a * sd_b], [rho_ab * sd_a * sd_b, sd_b * sd_b]], dtype=np.float64)
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_mc, n, 2))
    samples = z @ chol.T
    samples[:, :, 0] += mu_a
    samples[:, :, 1] += mu_b

    mean_a = samples[:, :, 0].mean(axis=1)
    sd_hat_a = samples[:, :, 0].std(axis=1, ddof=1)
    mean_b = samples[:, :, 1].mean(axis=1)
    sd_hat_b = samples[:, :, 1].std(axis=1, ddof=1)

    t_a = mean_a / (sd_hat_a / math.sqrt(n))
    t_b = (mean_b - eps_b) / (sd_hat_b / math.sqrt(n))
    crit_a = -t_ppf(1.0 - alpha_a, n - 1)
    crit_b = -t_ppf(1.0 - alpha_b, n - 1)

    rej_a = t_a < crit_a
    rej_b = t_b < crit_b
    rej_both = rej_a & rej_b
    return {
        "mc_power_a": float(np.mean(rej_a)),
        "mc_power_b": float(np.mean(rej_b)),
        "mc_power_joint": float(np.mean(rej_both)),
    }


def summarize_strata() -> List[dict]:
    bc_rows = load_csv(EXP_DIR / "robust_r3_budget_control" / "PAIRED_BUDGET_CONTROL.csv")
    diag_rows = load_csv(EXP_DIR / "robust_r3_selection_diagnostics" / "results" / "DIAG_PART1_PAIRED_AND_MC_SE.csv")
    lr_rows = load_csv(EXP_DIR / "robust_r3_lr_stability" / "results" / "run" / "LR_DEV_EVAL_FULL_PRECISION.csv")
    lr_idx = {(int(r["seed"]), r["ckpt_kind"], r["config"], r["scenario"]): r for r in lr_rows}

    records = []

    def add_stratum(stage, seeds_label, schedule, ckpt_kind, eval_paths, he_vals, oce_vals):
        he_arr = np.asarray(he_vals, dtype=np.float64)
        oce_arr = np.asarray(oce_vals, dtype=np.float64)
        records.append({
            "stage": stage,
            "seeds": seeds_label,
            "lr_schedule": schedule,
            "ckpt_kind": ckpt_kind,
            "eval_paths": eval_paths,
            "n_seeds": int(len(he_arr)),
            "dHE_030_mean": float(np.mean(he_arr)),
            "dHE_030_sd": float(np.std(he_arr, ddof=1)),
            "dHE_030_min": float(np.min(he_arr)),
            "dHE_030_max": float(np.max(he_arr)),
            "dOCE_020_mean": float(np.mean(oce_arr)),
            "dOCE_020_sd": float(np.std(oce_arr, ddof=1)),
            "dOCE_020_min": float(np.min(oce_arr)),
            "dOCE_020_max": float(np.max(oce_arr)),
        })

    for lbl, slist in [("BC_old_pilot", [3101, 3102, 3103]), ("BC_new_dev", [3201, 3202, 3203])]:
        he = [float(r["delta_he_net_ddof0"]) for r in bc_rows if r["split"] == "test" and r["comparison"] == "PRIMARY_R2m128_minus_Aref128" and r["scenario"] == "fixed_030" and int(r["seed"]) in slist]
        oce = [float(r["delta_hedged_oce_y_nom"]) for r in bc_rows if r["split"] == "test" and r["comparison"] == "PRIMARY_R2m128_minus_Aref128" and r["scenario"] == "fixed_020" and int(r["seed"]) in slist]
        add_stratum(lbl, str(slist), "FIXED", "best_val", 5000, he, oce)

    for dset, n_p in [("val_diag_a_20k", 20000), ("val_diag_c_20k", 20000), ("val_diag_b0_5k", 5000)]:
        for tag, ck in [("PRIMARY_BEST_R2_minus_A", "best_val"), ("PRIMARY_EP200_R2_minus_A", "ep200")]:
            he = [float(r["delta_he_net_ddof0"]) for r in diag_rows if r["comparison"] == tag and r["dataset"] == dset and r["scenario"] == "fixed_030"]
            oce = [float(r["delta_oce_y_frozen"]) for r in diag_rows if r["comparison"] == tag and r["dataset"] == dset and r["scenario"] == "fixed_020"]
            add_stratum(f"DIAG_{dset}", "[3101..3103, 3201..3203]", "FIXED", ck, n_p, he, oce)

    for sch in ("FIXED", "DECAY"):
        for ck in ("best_val", "ep200"):
            he = [float(lr_idx[(s, ck, f"R2_{sch}", "fixed_030")]["he_net_ddof0"]) - float(lr_idx[(s, ck, f"A_{sch}", "fixed_030")]["he_net_ddof0"]) for s in (3401, 3402, 3403)]
            oce = [float(lr_idx[(s, ck, f"R2_{sch}", "fixed_020")]["hedged_oce_y_nom"]) - float(lr_idx[(s, ck, f"A_{sch}", "fixed_020")]["hedged_oce_y_nom"]) for s in (3401, 3402, 3403)]
            add_stratum("LR_STAB", "[3401, 3402, 3403]", sch, ck, 5000, he, oce)

    return records


def main():
    eps_b = 0.00010
    strata = summarize_strata()
    (CONF_DIR / "dev_strata_summary.json").write_text(json.dumps(strata, indent=2), encoding="utf-8")

    # Sensitivity scenarios grounded in dev variance range WITHOUT pooling heterogeneous designs
    # Notice: DECAY 3-seed observed sd_B=0.000047, mu_B=-0.000002;
    # FIXED 6-seed 20k sd_B=0.000140, FIXED new_dev [3201..3203] sd_B=0.000069;
    # Hence we sweep plausible & stress SD_B from 0.000050 up to 0.000140, and mu_B from 0.000000 to +0.000040.
    regimes = [
        # (regime_id, note, mu_A, sd_A, mu_B, sd_B, rho_ab)
        ("R1_LR_DECAY_PointEst", "Near 3-seed LR_DECAY point estimate (optimistic variance)", -0.0048, 0.0008, 0.00000, 0.000050, 0.0),
        ("R2_Realistic_Base",    "Mild nominal degradation (+0.10 bp) & moderate SD_B=0.000065", -0.0038, 0.0012, 0.00001, 0.000065, 0.0),
        ("R3_Conservative_Design","Design assumption: +0.20 bp OCE cost, SD_B=0.000075 (1.6x 3-seed DECAY SD)", -0.0030, 0.0015, 0.00002, 0.000075, 0.0),
        ("R4_NegCorr_Conservative","Conservative design with adversarial rho(dHE, dOCE)=-0.30", -0.0030, 0.0015, 0.00002, 0.000075, -0.3),
        ("R5_PosCorr_Conservative","Conservative design with positive rho(dHE, dOCE)=+0.40", -0.0030, 0.0015, 0.00002, 0.000075, 0.4),
        ("R6_HighDrift_Stress",  "Small remaining margin (mu_B=+0.30 bp, margin=0.70 bp), SD_B=0.000085", -0.0025, 0.0018, 0.00003, 0.000085, 0.0),
        ("R7_HeavyVariance_FIXEDlike","Stress SD_B=0.000110 (closer to FIXED multi-seed dispersion)", -0.0030, 0.0025, 0.00002, 0.000110, 0.0),
        ("R8_Worst_40pctMargin_FIXEDsd","Extreme stress: mu_B=+0.40 bp (margin=0.60 bp) & SD_B=0.000120", -0.0020, 0.0025, 0.00004, 0.000120, -0.3),
    ]

    candidate_Ns = [12, 16, 20, 24, 30, 36]
    alpha_choices = [0.025, 0.05]

    csv_rows = []
    for alpha in alpha_choices:
        for reg_id, note, mu_a, sd_a, mu_b, sd_b, rho_ab in regimes:
            margin_b = eps_b - mu_b
            n80_b = next((n for n in range(4, 301) if nct_power_lower_tail(n, margin_b, sd_b, alpha) >= 0.80), None)
            n90_b = next((n for n in range(4, 301) if nct_power_lower_tail(n, margin_b, sd_b, alpha) >= 0.90), None)
            n90_a = next((n for n in range(4, 301) if nct_power_lower_tail(n, -mu_a, sd_a, alpha) >= 0.90), None)
            for n in candidate_Ns:
                pa = nct_power_lower_tail(n, -mu_a, sd_a, alpha)
                pb = nct_power_lower_tail(n, margin_b, sd_b, alpha)
                frechet_lb = max(0.0, pa + pb - 1.0)
                indep_joint = pa * pb
                mc_res = simulate_joint_iut_power(n, mu_a, sd_a, mu_b, sd_b, eps_b, rho_ab, alpha, alpha, n_mc=60000)
                csv_rows.append({
                    "alpha_one_sided": alpha,
                    "regime_id": reg_id,
                    "assumption_note": note,
                    "mu_A_dHE030": mu_a,
                    "sd_A_dHE030": sd_a,
                    "mu_B_dOCE020": mu_b,
                    "eps_B": eps_b,
                    "margin_B_eps_minus_muB": round(margin_b, 8),
                    "sd_B_dOCE020": sd_b,
                    "rho_AB": rho_ab,
                    "N_seeds": n,
                    "power_A_exact_nct": round(pa, 5),
                    "power_B_exact_nct": round(pb, 5),
                    "joint_power_frechet_lower_bound": round(frechet_lb, 5),
                    "joint_power_indep_approx": round(indep_joint, 5),
                    "joint_power_mc_bivariate": round(mc_res["mc_power_joint"], 5),
                    "N_req_A_90pct": n90_a,
                    "N_req_B_80pct": n80_b,
                    "N_req_B_90pct": n90_b,
                })

    out_csv = CONF_DIR / "SAMPLE_SIZE_SENSITIVITY.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print(f"Wrote {out_csv} ({len(csv_rows)} rows). Key Summary at alpha=0.025 (one-sided, aligned with 95% two-sided CI):")
    for reg_id, *_ in regimes:
        sub = [r for r in csv_rows if r["alpha_one_sided"] == 0.025 and r["regime_id"] == reg_id]
        r0 = sub[0]
        p_map = {r["N_seeds"]: r["joint_power_mc_bivariate"] for r in sub}
        print(
            f"  {reg_id:28s} | N90_A={str(r0['N_req_A_90pct']):>2s} N80_B={str(r0['N_req_B_80pct']):>3s} N90_B={str(r0['N_req_B_90pct']):>3s} "
            f"| Joint Power N=12:{p_map[12]:.3f} N=16:{p_map[16]:.3f} N=20:{p_map[20]:.3f} N=24:{p_map[24]:.3f} N=30:{p_map[30]:.3f} N=36:{p_map[36]:.3f}"
        )


if __name__ == "__main__":
    main()
