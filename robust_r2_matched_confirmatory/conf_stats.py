"""
conf_stats.py: Confirmatory paired statistical inference & audit diagnostics.

Implements:
- Completeness & validity ledger (missing seeds, partial pairs, non-finite detection).
- Co-primary paired t-tests aligned with two-sided 95% CIs under Berger IUT (alpha_one_sided=0.025):
    * H1a (HE_030 superiority): mu(Delta_HE_030) < 0.0
    * H1b (OCE_020 non-inferiority): mu(Delta_OCE_020) < +0.00010 (eps)
- Robustness & sensitivity diagnostics:
    * Exact binomial Sign test (null: P(d < bound) = 0.5; different null from mean t-test)
    * Wilcoxon signed-rank test (exact enumeration for n <= 16, continuity-corrected normal + tie adjustment for n > 16; null: symmetric around bound)
    * Skewness / excess kurtosis & leave-one-out (LOO) mean range & decision flip check.
- Secondary scenario & mechanism metrics summary (never gated as PASS/FAIL).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

CONF_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CONF_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sample_size_and_power import t_cdf, t_ppf


def exact_binom_lower_p(k_success: int, n_trials: int, p0: float = 0.5) -> float:
    """One-sided exact binomial p-value: P(X >= k_success) under Bin(n_trials, p0)."""
    if n_trials <= 0:
        return 1.0
    total = 0.0
    for k in range(k_success, n_trials + 1):
        total += math.comb(n_trials, k) * (p0 ** k) * ((1.0 - p0) ** (n_trials - k))
    return float(min(1.0, max(0.0, total)))


def _rank_abs_with_ties(abs_vals: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute average ranks (1..m) and tie-correction term sum(t^3 - t)."""
    m = len(abs_vals)
    order = np.argsort(abs_vals, kind="mergesort")
    ranks = np.empty(m, dtype=np.float64)
    tie_cubed_sum = 0.0
    i = 0
    while i < m:
        j = i + 1
        while j < m and abs_vals[order[j]] == abs_vals[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        t_len = j - i
        if t_len > 1:
            tie_cubed_sum += float(t_len ** 3 - t_len)
        for idx in range(i, j):
            ranks[order[idx]] = avg_rank
        i = j
    return ranks, tie_cubed_sum


def wilcoxon_lower_tail_p(
    shifted_vals: Sequence[float], exact_max_m: int = 50
) -> Dict[str, Any]:
    """
    One-sided Wilcoxon signed-rank p-value testing shifted_vals < 0 (lower tail).
    Uses Wilcoxon zero-excluded convention (drops exact zeros prior to ranking) and reports nonzero count m.
    """
    arr = np.asarray(shifted_vals, dtype=np.float64)
    nz = arr[arr != 0.0]
    m = int(len(nz))
    if m == 0:
        return {"m_nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "p_one_sided": 1.0, "method": "degenerate_all_zero"}

    ranks, tie_sum = _rank_abs_with_ties(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0.0]))
    w_minus = float(np.sum(ranks[nz < 0.0]))
    # Lower tail alternative (vals < 0) corresponds to SMALL W+ (or equivalently large W-)
    if m <= exact_max_m and tie_sum == 0.0:
        # Exact enumeration over 2^m sign assignments
        # Count assignments with sum(ranks_pos) <= w_plus + 1e-12
        r_ints = [int(round(r)) for r in ranks]
        max_sum = sum(r_ints)
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        for r in r_ints:
            for s in range(max_sum, r - 1, -1):
                dp[s] += dp[s - r]
        target = int(math.floor(w_plus + 1e-9))
        p_val = sum(dp[: target + 1]) / float(1 << m)
        return {
            "m_nonzero": m,
            "w_plus": w_plus,
            "w_minus": w_minus,
            "p_one_sided": float(min(1.0, max(0.0, p_val))),
            "method": "exact_enumeration",
        }

    mean_w = m * (m + 1.0) / 4.0
    var_w = (m * (m + 1.0) * (2.0 * m + 1.0) - 0.5 * tie_sum) / 24.0
    if var_w <= 0.0:
        return {"m_nonzero": m, "w_plus": w_plus, "w_minus": w_minus, "p_one_sided": 1.0, "method": "zero_variance"}
    # Lower-tail cumulative probability P(W+ <= w_plus) uses +0.5 continuity correction
    cc = 0.5
    z = (w_plus - mean_w + cc) / math.sqrt(var_w)
    p_val = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return {
        "m_nonzero": m,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_one_sided": float(min(1.0, max(0.0, p_val))),
        "method": "normal_approx_continuity_tie_adjusted",
    }


def paired_t_summary(
    diffs: Sequence[float],
    bound: float = 0.0,
    alpha_one_sided: float = 0.025,
) -> Dict[str, Any]:
    """
    Paired lower-tail test: H0: mu >= bound vs H1: mu < bound.
    Uses two-sided (1 - 2 * alpha_one_sided) CI so CI95_high < bound <-> p_one_sided < alpha_one_sided.
    """
    arr = np.asarray(diffs, dtype=np.float64)
    n = int(len(arr))
    if n < 2 or not np.all(np.isfinite(arr)):
        return {
            "n": n,
            "finite": bool(np.all(np.isfinite(arr))) if n > 0 else False,
            "pass_bound": False,
            "error": "insufficient_or_non_finite_pairs",
        }

    mean_d = float(np.mean(arr))
    sd_d = float(np.std(arr, ddof=1))
    se_d = float(sd_d / math.sqrt(n))
    df = n - 1
    t_crit = float(t_ppf(1.0 - alpha_one_sided, df))

    if se_d == 0.0:
        if mean_d < bound:
            t_stat = -math.inf
            p_one = 0.0
            pass_bound = True
        else:
            t_stat = math.inf if mean_d > bound else 0.0
            p_one = 1.0 if mean_d >= bound else 0.0
            pass_bound = False
    else:
        t_stat = float((mean_d - bound) / se_d)
        p_one = float(t_cdf(t_stat, df))
        pass_bound = bool((mean_d + t_crit * se_d) < bound)

    ci_low = float(mean_d - t_crit * se_d)
    ci_high = float(mean_d + t_crit * se_d)

    # Higher moments (sample skewness & excess kurtosis)
    centered = arr - mean_d
    if sd_d > 0.0:
        m3 = float(np.mean(centered ** 3))
        m4 = float(np.mean(centered ** 4))
        s_pop = float(np.std(arr, ddof=0))
        skew = m3 / (s_pop ** 3) if s_pop > 0.0 else 0.0
        kurt_ex = m4 / (s_pop ** 4) - 3.0 if s_pop > 0.0 else 0.0
    else:
        skew = 0.0
        kurt_ex = 0.0

    # Leave-one-out influence & decision stability
    loo_means: List[float] = []
    loo_flips = 0
    if n >= 3:
        t_crit_loo = float(t_ppf(1.0 - alpha_one_sided, n - 2))
        for i in range(n):
            sub = np.delete(arr, i)
            m_sub = float(np.mean(sub))
            sd_sub = float(np.std(sub, ddof=1))
            se_sub = sd_sub / math.sqrt(n - 1)
            loo_pass = bool((m_sub + t_crit_loo * se_sub) < bound)
            if loo_pass != pass_bound:
                loo_flips += 1
            loo_means.append(m_sub)
    else:
        loo_means = [mean_d]

    # Supportive non-parametric checks (distinct nulls)
    shifted = arr - bound
    n_success = int(np.sum(shifted < 0.0))
    n_sign_ties = int(np.sum(shifted == 0.0))
    sign_p = exact_binom_lower_p(n_success, n - n_sign_ties, 0.5) if (n - n_sign_ties) > 0 else 1.0
    wilcox = wilcoxon_lower_tail_p(shifted)

    q25, q50, q75 = [float(x) for x in np.percentile(arr, [25.0, 50.0, 75.0])]
    return {
        "n": n,
        "df": df,
        "finite": True,
        "bound": float(bound),
        "alpha_one_sided": float(alpha_one_sided),
        "ci_two_sided_level": float(1.0 - 2.0 * alpha_one_sided),
        "mean": mean_d,
        "sd_ddof1": sd_d,
        "se": se_d,
        "t_stat": t_stat,
        "p_one_sided": p_one,
        "t_crit": t_crit,
        "ci95_two_sided_low": ci_low,
        "ci95_two_sided_high": ci_high,
        "one_sided_ucl": ci_high,
        "pass_bound": pass_bound,
        "median": q50,
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "skewness": float(skew),
        "excess_kurtosis": float(kurt_ex),
        "loo_mean_min": float(min(loo_means)),
        "loo_mean_max": float(max(loo_means)),
        "loo_decision_flips": int(loo_flips),
        "supportive_sign_n_below_bound": n_success,
        "supportive_sign_ties": n_sign_ties,
        "supportive_sign_p_one_sided": float(sign_p),
        "supportive_wilcoxon": wilcox,
    }


def _extract_valid_pairs(
    paired_rows: Sequence[Dict[str, Any]],
    planned_seeds: Sequence[int],
    scenario: str,
    ckpt_kind: str,
    metric_key_diff: str,
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """Build completeness & non-finite ledger before computing any statistic."""
    by_seed: Dict[int, Dict[str, Any]] = {}
    duplicate_seeds: List[int] = []
    for row in paired_rows:
        if row.get("scenario") != scenario or row.get("ckpt_kind") != ckpt_kind:
            continue
        s = int(row["seed"])
        if s in by_seed:
            duplicate_seeds.append(s)
        by_seed[s] = row

    missing_seeds: List[int] = []
    non_finite_seeds: List[int] = []
    valid_seeds: List[int] = []
    valid_diffs: List[float] = []

    for s in planned_seeds:
        if s not in by_seed:
            missing_seeds.append(int(s))
            continue
        val = by_seed[s].get(metric_key_diff)
        try:
            fval = float(val)
        except (TypeError, ValueError):
            non_finite_seeds.append(int(s))
            continue
        if not math.isfinite(fval):
            non_finite_seeds.append(int(s))
            continue
        valid_seeds.append(int(s))
        valid_diffs.append(fval)

    unexpected_seeds = sorted(set(by_seed.keys()) - set(int(s) for s in planned_seeds))
    ledger = {
        "scenario": scenario,
        "ckpt_kind": ckpt_kind,
        "metric": metric_key_diff,
        "planned_n": len(planned_seeds),
        "valid_n": len(valid_seeds),
        "missing_seeds": missing_seeds,
        "non_finite_seeds": non_finite_seeds,
        "duplicate_seeds": sorted(set(duplicate_seeds)),
        "unexpected_seeds": unexpected_seeds,
        "complete_and_clean": (
            len(valid_seeds) == len(planned_seeds)
            and len(missing_seeds) == 0
            and len(non_finite_seeds) == 0
            and len(duplicate_seeds) == 0
            and len(unexpected_seeds) == 0
        ),
    }
    return valid_seeds, valid_diffs, ledger


def analyze_confirmatory_pairs(
    paired_rows: Sequence[Dict[str, Any]],
    planned_seeds: Sequence[int],
    eps_margin: float = 0.00010,
    alpha_one_sided: float = 0.025,
    require_complete_clean: bool = True,
) -> Dict[str, Any]:
    """
    Execute the locked Berger Intersection-Union Test (IUT) on confirmatory paired rows.

    Expected fields per row in paired_rows (matching (seed, ckpt_kind, scenario)):
      - delta_he_R2_minus_A
      - delta_oce_R2_minus_A
    Plus optional secondary paired diffs:
      - delta_cvar95_R2_minus_A
      - delta_turnover_R2_minus_A
    """
    _, d_he_030, ledger_a = _extract_valid_pairs(
        paired_rows, planned_seeds, "fixed_030", "best_val", "delta_he_R2_minus_A"
    )
    _, d_oce_020, ledger_b = _extract_valid_pairs(
        paired_rows, planned_seeds, "fixed_020", "best_val", "delta_oce_R2_minus_A"
    )

    h1a = paired_t_summary(d_he_030, bound=0.0, alpha_one_sided=alpha_one_sided)
    h1b = paired_t_summary(d_oce_020, bound=eps_margin, alpha_one_sided=alpha_one_sided)

    complete_gate = ledger_a["complete_and_clean"] and ledger_b["complete_and_clean"]
    raw_iut_pass = bool(h1a.get("pass_bound", False) and h1b.get("pass_bound", False))

    if require_complete_clean and not complete_gate:
        verdict = "INCOMPLETE_OR_FAILED_AUDIT"
        confirmatory_pass = False
    elif raw_iut_pass:
        verdict = "PASS_CONFIRMATORY_IUT"
        confirmatory_pass = True
    else:
        verdict = "FAIL_CONFIRMATORY_IUT"
        confirmatory_pass = False

    p_iut = max(
        float(h1a.get("p_one_sided", 1.0)),
        float(h1b.get("p_one_sided", 1.0)),
    )

    # Secondary scenarios & ep200 sensitivity summaries (descriptive only; never override IUT)
    secondary_summaries: List[Dict[str, Any]] = []
    present_combos = sorted(
        {(str(r["ckpt_kind"]), str(r["scenario"])) for r in paired_rows}
    )
    for ckpt_kind, scen in present_combos:
        for metric_key in (
            "A_he_net_ddof0",
            "R2_he_net_ddof0",
            "delta_he_R2_minus_A",
            "A_hedged_oce_y_nom",
            "R2_hedged_oce_y_nom",
            "delta_oce_R2_minus_A",
            "R2_hedged_oce_y_scen",
            "delta_oce_y_scen_R2_minus_A",
            "A_cvar95",
            "R2_cvar95",
            "delta_cvar95_R2_minus_A",
            "A_turnover",
            "R2_turnover",
            "delta_turnover_R2_minus_A",
        ):
            scen_has_finite = any(
                r.get("scenario") == scen and r.get("ckpt_kind") == ckpt_kind and r.get(metric_key) is not None
                for r in paired_rows
            )
            if scen_has_finite:
                _, vals, ldg = _extract_valid_pairs(
                    paired_rows, planned_seeds, scen, ckpt_kind, metric_key
                )
                if len(vals) >= 2:
                    st = paired_t_summary(vals, bound=0.0, alpha_one_sided=alpha_one_sided)
                    secondary_summaries.append(
                        {
                            "ckpt_kind": ckpt_kind,
                            "scenario": scen,
                            "metric": metric_key,
                            "valid_n": ldg["valid_n"],
                            "planned_n": ldg["planned_n"],
                            "mean": st["mean"],
                            "sd_ddof1": st["sd_ddof1"],
                            "se": st["se"],
                            "ci95_low": st["ci95_two_sided_low"],
                            "ci95_high": st["ci95_two_sided_high"],
                            "median": st["median"],
                            "min": st["min"],
                            "max": st["max"],
                        }
                    )

    return {
        "verdict": verdict,
        "confirmatory_pass": confirmatory_pass,
        "raw_iut_pass_before_completeness_gate": raw_iut_pass,
        "complete_and_clean_gate": complete_gate,
        "p_iut_max": p_iut,
        "alpha_one_sided": float(alpha_one_sided),
        "eps_margin": float(eps_margin),
        "ledger": {
            "H1a_HE_030": ledger_a,
            "H1b_OCE_020": ledger_b,
        },
        "co_primary": {
            "H1a_HE_030_best_val": h1a,
            "H1b_OCE_020_best_val": h1b,
        },
        "secondary_summaries": secondary_summaries,
        "methodological_note": (
            "Primary decision strictly follows Berger Intersection-Union Test (IUT) on paired "
            "parametric t-tests at one-sided alpha=0.025 (two-sided 95% CI upper bound < 0 for H1a "
            "and < +0.00010 for H1b). Sign and Wilcoxon tests assess different nulls (median=bound "
            "and symmetry around bound) and are reported strictly as robustness diagnostics."
        ),
    }


def load_and_merge_paired_csvs(csv_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Load one or more PAIRED_CONFIRMATORY.csv slice files for standalone IUT evaluation."""
    import csv

    merged: List[Dict[str, Any]] = []
    for cp in csv_paths:
        with open(cp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged.append(dict(row))
    return merged


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Standalone confirmatory Berger IUT analyzer")
    parser.add_argument("--paired-csv", nargs="+", required=True, help="One or more PAIRED_CONFIRMATORY.csv files")
    parser.add_argument("--seed-start", type=int, default=4001)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--eps", type=float, default=0.00010)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    planned = list(range(args.seed_start, args.seed_start + args.seed_count))
    rows = load_and_merge_paired_csvs([Path(p) for p in args.paired_csv])
    res = analyze_confirmatory_pairs(
        rows,
        planned_seeds=planned,
        eps_margin=args.eps,
        alpha_one_sided=args.alpha,
        require_complete_clean=True,
    )
    txt = json.dumps(res, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(txt, encoding="utf-8")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
