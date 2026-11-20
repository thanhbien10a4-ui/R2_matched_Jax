"""
robust_r2_matched_cost_confirmatory/cost_stats.py:
Confirmatory Berger IUT inference + completeness ledger + expanded net/gross/cost/mechanism
secondary summaries. Reuses core statistical routines from robust_r2_matched_confirmatory/conf_stats.py.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

COST_CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = COST_CONF_DIR.parent
R2_CONF_DIR = EXP_DIR / "robust_r2_matched_confirmatory"

for _p in (str(R2_CONF_DIR), str(COST_CONF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from conf_stats import (
    _extract_valid_pairs,
    exact_binom_lower_p,
    paired_t_summary,
    wilcoxon_lower_tail_p,
)
from cost_design import ALPHA_ONE_SIDED, EPS_NONINFERIORITY_OCE020
from cost_keys import COST_LOCKED_COHORT_SEEDS

SECONDARY_METRIC_KEYS = (
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
    "A_var95",
    "R2_var95",
    "delta_var95_R2_minus_A",
    "A_mean_gains",
    "R2_mean_gains",
    "delta_mean_gains_R2_minus_A",
    "A_mean_gross_gains",
    "R2_mean_gross_gains",
    "delta_mean_gross_gains_R2_minus_A",
    "A_mean_cost",
    "R2_mean_cost",
    "delta_mean_cost_R2_minus_A",
    "A_turnover",
    "R2_turnover",
    "delta_turnover_R2_minus_A",
    "A_active_steps_gt1e5",
    "R2_active_steps_gt1e5",
    "delta_active_steps_R2_minus_A",
    "A_immobility_rate",
    "R2_immobility_rate",
    "delta_immobility_rate_R2_minus_A",
    "A_trade_size_active_pooled",
    "R2_trade_size_active_pooled",
    "delta_trade_size_active_pooled_R2_minus_A",
    "A_trade_size_active_pathwise",
    "R2_trade_size_active_pathwise",
    "delta_trade_size_active_pathwise_R2_minus_A",
    "A_max_abs_delta",
    "R2_max_abs_delta",
    "delta_max_abs_delta_R2_minus_A",
    "A_p99_abs_delta",
    "R2_p99_abs_delta",
    "delta_p99_abs_delta_R2_minus_A",
    "A_frac_steps_abs_delta_gt2",
    "R2_frac_steps_abs_delta_gt2",
    "delta_frac_steps_abs_delta_gt2_R2_minus_A",
)


def analyze_cost_confirmatory_pairs(
    paired_rows: Sequence[Dict[str, Any]],
    planned_seeds: Sequence[int] = tuple(COST_LOCKED_COHORT_SEEDS),
    eps_margin: float = EPS_NONINFERIORITY_OCE020,
    alpha_one_sided: float = ALPHA_ONE_SIDED,
    require_complete_clean: bool = True,
) -> Dict[str, Any]:
    _, d_he_030, ledger_a = _extract_valid_pairs(
        paired_rows, planned_seeds, "fixed_030", "best_val", "delta_he_R2_minus_A"
    )
    _, d_oce_020, ledger_b = _extract_valid_pairs(
        paired_rows, planned_seeds, "fixed_020", "best_val", "delta_oce_R2_minus_A"
    )

    h1a = paired_t_summary(d_he_030, bound=0.0, alpha_one_sided=alpha_one_sided)
    h1b = paired_t_summary(d_oce_020, bound=eps_margin, alpha_one_sided=alpha_one_sided)

    complete_gate = bool(ledger_a["complete_and_clean"] and ledger_b["complete_and_clean"])
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

    p_iut = max(float(h1a.get("p_one_sided", 1.0)), float(h1b.get("p_one_sided", 1.0)))

    secondary_summaries: List[Dict[str, Any]] = []
    present_combos = sorted({(str(r["ckpt_kind"]), str(r["scenario"])) for r in paired_rows})
    for ckpt_kind, scen in present_combos:
        for metric_key in SECONDARY_METRIC_KEYS:
            _, vals, ldg = _extract_valid_pairs(paired_rows, planned_seeds, scen, ckpt_kind, metric_key)
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
            "Primary decision strictly follows Berger Intersection-Union Test (IUT) on paired parametric "
            "t-tests at one-sided alpha=0.025 (two-sided 95% CI upper bound < 0 for H1a net HE at sigma=0.30 "
            "and < +0.00010 for H1b net OCE at sigma=0.20, all evaluated under c_s=0.0010)."
        ),
    }


def load_paired_csvs(csv_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for cp in csv_paths:
        with open(cp, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merged.append(dict(row))
    return merged


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Berger IUT analyzer for 10 bps cost confirmatory runs")
    parser.add_argument("--paired-csv", nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, default=5001)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--eps", type=float, default=EPS_NONINFERIORITY_OCE020)
    parser.add_argument("--alpha", type=float, default=ALPHA_ONE_SIDED)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    planned = list(range(args.seed_start, args.seed_start + args.seed_count))
    rows = load_paired_csvs([Path(p) for p in args.paired_csv])
    res = analyze_cost_confirmatory_pairs(
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
