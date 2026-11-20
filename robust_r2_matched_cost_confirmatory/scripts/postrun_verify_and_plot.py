"""
robust_r2_matched_cost_confirmatory/scripts/postrun_verify_and_plot.py:
Independent post-run audit verifier & publication plot generator for Job 16914 (c_s = 0.0010, N=24).
Never touches HOLD_EVAL or [5025..5040].
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
COST_CONF_DIR = SCRIPTS_DIR.parent
EXP_DIR = COST_CONF_DIR.parent
R2_CONF_DIR = EXP_DIR / "robust_r2_matched_confirmatory"

for _p in (str(COST_CONF_DIR), str(R2_CONF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cost_stats import analyze_cost_confirmatory_pairs, load_paired_csvs, paired_t_summary
from cost_keys import COST_LOCKED_COHORT_SEEDS, COST_EVAL_SCENARIOS


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_dataset_hash_isolation(summary: Dict[str, Any]) -> Dict[str, Any]:
    dhashes = summary.get("dataset_hashes", {})
    all_hashes: List[str] = []
    val_hashes_by_seed: Dict[int, Dict[str, str]] = {}
    train_hashes_by_seed: Dict[int, Dict[str, str]] = {}
    test_hashes_by_seed: Dict[int, Dict[str, str]] = {}

    for s in COST_LOCKED_COHORT_SEEDS:
        entry = dhashes.get(str(s), {})
        tr = entry.get("train", {})
        va = entry.get("val", {})
        te = entry.get("conf_test", {})
        train_hashes_by_seed[s] = tr
        val_hashes_by_seed[s] = va
        test_hashes_by_seed[s] = te
        all_hashes.extend(tr.values())
        all_hashes.extend(va.values())
        all_hashes.extend(te.values())

    # Check Val(s) vs Train(s+1) hash overlap + across-seed overlap
    adj_val_train_overlap = []
    for i in range(len(COST_LOCKED_COHORT_SEEDS) - 1):
        s1, s2 = COST_LOCKED_COHORT_SEEDS[i], COST_LOCKED_COHORT_SEEDS[i + 1]
        v_set = set(val_hashes_by_seed[s1].values())
        t_set = set(train_hashes_by_seed[s2].values())
        inter = v_set.intersection(t_set)
        if inter:
            adj_val_train_overlap.append((s1, s2, sorted(inter)))

    # Across distinct seeds (train, val, test within each pool)
    cross_seed_collisions = len(all_hashes) - len(set(all_hashes))
    return {
        "n_seeds_hashed": len(dhashes),
        "total_world_hashes": len(all_hashes),
        "unique_world_hashes": len(set(all_hashes)),
        "exact_cross_world_hash_collisions": cross_seed_collisions,
        "adjacent_val_train_hash_overlaps": adj_val_train_overlap,
    }


def verify_source_sha_parity(summary: Dict[str, Any], source_sha_file: Path) -> Dict[str, Any]:
    disk_map: Dict[str, str] = {}
    if source_sha_file.exists():
        for line in source_sha_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            norm_rel = rel.strip()
            if norm_rel.startswith("../"):
                norm_rel = norm_rel[3:]
            else:
                norm_rel = f"robust_r2_matched_cost_confirmatory/{norm_rel}"
            disk_map[norm_rel] = digest.strip()

    run_map = summary.get("source_sha256", {})
    common_keys = sorted(set(disk_map.keys()) & set(run_map.keys()))
    mismatches = [k for k in common_keys if disk_map[k] != run_map[k]]
    return {
        "source_sha_file_entries": len(disk_map),
        "summary_source_sha_entries": len(run_map),
        "common_verified": len(common_keys),
        "mismatches": mismatches,
        "all_common_match": len(mismatches) == 0 and len(common_keys) >= 19,
    }


def extract_seed_table_and_summaries(
    eval_rows: List[Dict[str, str]],
    paired_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    eval_idx = {(int(r["seed"]), r["ckpt_kind"], r["scenario"], r["config"]): r for r in eval_rows}
    pair_idx = {(int(r["seed"]), r["ckpt_kind"], r["scenario"]): r for r in paired_rows}

    per_seed_rows: List[Dict[str, Any]] = []
    for s in COST_LOCKED_COHORT_SEEDS:
        r030_b = pair_idx[(s, "best_val", "fixed_030")]
        r020_b = pair_idx[(s, "best_val", "fixed_020")]
        r030_e = pair_idx[(s, "ep200", "fixed_030")]
        r020_e = pair_idx[(s, "ep200", "fixed_020")]
        ea_020 = eval_idx[(s, "best_val", "fixed_020", "A_DECAY")]
        er_020 = eval_idx[(s, "best_val", "fixed_020", "R2_DECAY")]
        per_seed_rows.append(
            {
                "seed": s,
                "A_best_epoch": int(ea_020["best_epoch"]),
                "R2_best_epoch": int(er_020["best_epoch"]),
                "A_he_030_best": float(r030_b["A_he_net_ddof0"]),
                "R2_he_030_best": float(r030_b["R2_he_net_ddof0"]),
                "dHE_030_best": float(r030_b["delta_he_R2_minus_A"]),
                "A_oce_020_best": float(r020_b["A_hedged_oce_y_nom"]),
                "R2_oce_020_best": float(r020_b["R2_hedged_oce_y_nom"]),
                "dOCE_020_best": float(r020_b["delta_oce_R2_minus_A"]),
                "dHE_030_ep200": float(r030_e["delta_he_R2_minus_A"]),
                "dOCE_020_ep200": float(r020_e["delta_oce_R2_minus_A"]),
                "A_cost_020_best": float(r020_b["A_mean_cost"]),
                "R2_cost_020_best": float(r020_b["R2_mean_cost"]),
                "dCost_020_best": float(r020_b["delta_mean_cost_R2_minus_A"]),
                "A_turn_020_best": float(r020_b["A_turnover"]),
                "R2_turn_020_best": float(r020_b["R2_turnover"]),
                "A_immob_020_best": float(r020_b["A_immobility_rate"]),
                "R2_immob_020_best": float(r020_b["R2_immobility_rate"]),
            }
        )

    # Full 6-scenario summary table for both best_val and ep200
    scen_table: List[Dict[str, Any]] = []
    metrics_to_agg = [
        ("he_net_ddof0", "A_he_net_ddof0", "R2_he_net_ddof0", "delta_he_R2_minus_A"),
        ("hedged_oce_y_nom", "A_hedged_oce_y_nom", "R2_hedged_oce_y_nom", "delta_oce_R2_minus_A"),
        ("cvar95", "A_cvar95", "R2_cvar95", "delta_cvar95_R2_minus_A"),
        ("var95", "A_var95", "R2_var95", "delta_var95_R2_minus_A"),
        ("mean_gains", "A_mean_gains", "R2_mean_gains", "delta_mean_gains_R2_minus_A"),
        ("mean_gross_gains", "A_mean_gross_gains", "R2_mean_gross_gains", "delta_mean_gross_gains_R2_minus_A"),
        ("mean_cost", "A_mean_cost", "R2_mean_cost", "delta_mean_cost_R2_minus_A"),
        ("turnover", "A_turnover", "R2_turnover", "delta_turnover_R2_minus_A"),
        ("active_steps_gt1e5", "A_active_steps_gt1e5", "R2_active_steps_gt1e5", "delta_active_steps_R2_minus_A"),
        ("immobility_rate", "A_immobility_rate", "R2_immobility_rate", "delta_immobility_rate_R2_minus_A"),
        ("trade_size_active_pooled", "A_trade_size_active_pooled", "R2_trade_size_active_pooled", "delta_trade_size_active_pooled_R2_minus_A"),
        ("trade_size_active_pathwise", "A_trade_size_active_pathwise", "R2_trade_size_active_pathwise", "delta_trade_size_active_pathwise_R2_minus_A"),
        ("max_abs_delta", "A_max_abs_delta", "R2_max_abs_delta", "delta_max_abs_delta_R2_minus_A"),
        ("p99_abs_delta", "A_p99_abs_delta", "R2_p99_abs_delta", "delta_p99_abs_delta_R2_minus_A"),
        ("frac_steps_abs_delta_gt2", "A_frac_steps_abs_delta_gt2", "R2_frac_steps_abs_delta_gt2", "delta_frac_steps_abs_delta_gt2_R2_minus_A"),
    ]
    for ck in ("best_val", "ep200"):
        for sc in COST_EVAL_SCENARIOS:
            for mlabel, ka, kr2, kd in metrics_to_agg:
                va = np.array([float(pair_idx[(s, ck, sc)][ka]) for s in COST_LOCKED_COHORT_SEEDS], dtype=np.float64)
                vr2 = np.array([float(pair_idx[(s, ck, sc)][kr2]) for s in COST_LOCKED_COHORT_SEEDS], dtype=np.float64)
                vd = np.array([float(pair_idx[(s, ck, sc)][kd]) for s in COST_LOCKED_COHORT_SEEDS], dtype=np.float64)
                st = paired_t_summary(vd, bound=0.0, alpha_one_sided=0.025)
                scen_table.append(
                    {
                        "ckpt_kind": ck,
                        "scenario": sc,
                        "metric": mlabel,
                        "A_mean": float(np.mean(va)),
                        "A_sd": float(np.std(va, ddof=1)),
                        "R2_mean": float(np.mean(vr2)),
                        "R2_sd": float(np.std(vr2, ddof=1)),
                        "delta_mean": st["mean"],
                        "delta_sd": st["sd_ddof1"],
                        "delta_se": st["se"],
                        "ci95_low": st["ci95_two_sided_low"],
                        "ci95_high": st["ci95_two_sided_high"],
                        "n_lt_0": int(np.sum(vd < 0.0)),
                        "n_gt_0": int(np.sum(vd > 0.0)),
                    }
                )

    max_acct_resid = max(float(r["cost_accounting_residual"]) for r in eval_rows)
    zero_cost_rows = [r for r in eval_rows if float(r["mean_cost"]) <= 0.0]

    # Joint correlation rho_AB on co-primary endpoints
    d_he_030 = np.array([r["dHE_030_best"] for r in per_seed_rows])
    d_oce_020 = np.array([r["dOCE_020_best"] for r in per_seed_rows])
    rho_ab = float(np.corrcoef(d_he_030, d_oce_020)[0, 1])

    # ep200 sensitivity tests on co-primary endpoints
    ep200_h1a = paired_t_summary([r["dHE_030_ep200"] for r in per_seed_rows], bound=0.0, alpha_one_sided=0.025)
    ep200_h1b = paired_t_summary([r["dOCE_020_ep200"] for r in per_seed_rows], bound=0.00010, alpha_one_sided=0.025)

    # Extra diagnostics from eval_rows (immobility thresholds lt1e4, lt1e3, frac_ge45, zero_active_path_frac, actual steps)
    extra_mech: Dict[str, Any] = {}
    for ck in ("best_val", "ep200"):
        extra_mech[ck] = {}
        for sc in COST_EVAL_SCENARIOS:
            a_rows = [eval_idx[(s, ck, sc, "A_DECAY")] for s in COST_LOCKED_COHORT_SEEDS]
            r_rows = [eval_idx[(s, ck, sc, "R2_DECAY")] for s in COST_LOCKED_COHORT_SEEDS]
            extra_mech[ck][sc] = {
                "A_immob_lt1e5": float(np.mean([float(x["immobility_rate"]) for x in a_rows])),
                "R2_immob_lt1e5": float(np.mean([float(x["immobility_rate"]) for x in r_rows])),
                "A_immob_lt1e4": float(np.mean([float(x["immobility_rate_lt1e4"]) for x in a_rows])),
                "R2_immob_lt1e4": float(np.mean([float(x["immobility_rate_lt1e4"]) for x in r_rows])),
                "A_immob_lt1e3": float(np.mean([float(x["immobility_rate_lt1e3"]) for x in a_rows])),
                "R2_immob_lt1e3": float(np.mean([float(x["immobility_rate_lt1e3"]) for x in r_rows])),
                "A_zero_active_path_frac": float(np.mean([float(x["zero_active_path_frac"]) for x in a_rows])),
                "R2_zero_active_path_frac": float(np.mean([float(x["zero_active_path_frac"]) for x in r_rows])),
                "A_frac_delta_ge45": float(np.mean([float(x["frac_steps_abs_delta_ge45"]) for x in a_rows])),
                "R2_frac_delta_ge45": float(np.mean([float(x["frac_steps_abs_delta_ge45"]) for x in r_rows])),
            }

    return {
        "per_seed_rows": per_seed_rows,
        "scen_table": scen_table,
        "extra_mech": extra_mech,
        "rho_ab_best_val": rho_ab,
        "ep200_sensitivity": {"H1a_HE_030": ep200_h1a, "H1b_OCE_020": ep200_h1b},
        "max_cost_accounting_residual": max_acct_resid,
        "zero_cost_eval_rows_count": len(zero_cost_rows),
        "actual_steps_unique": sorted({int(r["actual_update_steps"]) for r in eval_rows}),
        "actual_paths_unique": sorted({int(r["actual_paths_processed"]) for r in eval_rows}),
    }


def make_plots(out_dir: Path, plots_dir: Path, tables: Dict[str, Any]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    rows = tables["per_seed_rows"]
    seeds = np.array([r["seed"] for r in rows])
    d_he = np.array([r["dHE_030_best"] for r in rows])
    d_oce = np.array([r["dOCE_020_best"] for r in rows])
    d_he_e = np.array([r["dHE_030_ep200"] for r in rows])
    d_oce_e = np.array([r["dOCE_020_ep200"] for r in rows])

    # Fig 1: Co-primary paired scatter / distributions + zero-cost benchmark line
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    cols_a = ["#1b9e77" if v < 0 else "#d95f02" for v in d_he]
    axes[0].bar(np.arange(len(seeds)), d_he * 1e4, color=cols_a, edgecolor="black", linewidth=0.4)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.1, label="Null boundary (0 bps)")
    m_he = np.mean(d_he) * 1e4
    axes[0].axhline(m_he, color="#08519c", linewidth=1.8, label=f"Mean = {m_he:+.2f} bps")
    axes[0].set_xticks(np.arange(len(seeds)))
    axes[0].set_xticklabels([str(s)[-2:] for s in seeds], fontsize=8)
    axes[0].set_xlabel("Confirmatory Seed [50xx]")
    axes[0].set_ylabel("Delta HE at sigma=0.30 (bps, x1e-4)")
    axes[0].set_title("H1a: Net Hedging Error Change at sigma=0.30 (10 bps Cost)")
    axes[0].legend(loc="lower right", fontsize=8.5)

    cols_b = ["#2b83ba" if v < 0.00010 else "#d7191c" for v in d_oce]
    axes[1].bar(np.arange(len(seeds)), d_oce * 1e4, color=cols_b, edgecolor="black", linewidth=0.4)
    axes[1].axhline(1.0, color="#d7191c", linestyle="--", linewidth=1.3, label="Non-inferiority eps = +1.00 bp")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0, label="Parity (0.00 bp)")
    m_oce = np.mean(d_oce) * 1e4
    axes[1].axhline(m_oce, color="#08519c", linewidth=1.8, label=f"Mean = {m_oce:+.2f} bps")
    axes[1].set_xticks(np.arange(len(seeds)))
    axes[1].set_xticklabels([str(s)[-2:] for s in seeds], fontsize=8)
    axes[1].set_xlabel("Confirmatory Seed [50xx]")
    axes[1].set_ylabel("Delta OCE y_nom at sigma=0.20 (bps, x1e-4)")
    axes[1].set_title("H1b: Net OCE Change at sigma=0.20 (10 bps Cost)")
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    f1 = plots_dir / "fig1_cost10_primary_paired_n24.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # Fig 2: 6-scenario best_val overview (Delta HE, Delta OCE_y_nom, Delta CVaR95, Delta Cost)
    stbl = {(r["ckpt_kind"], r["scenario"], r["metric"]): r for r in tables["scen_table"]}
    sc_names = list(COST_EVAL_SCENARIOS)
    short_sc = ["0.15", "0.20", "0.25", "rs15-25", "rs15-28", "0.30"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), dpi=150)
    specs = [
        ("he_net_ddof0", "Delta Net HE (bps, R2 - A)", 1e4),
        ("hedged_oce_y_nom", "Delta Net OCE_y_nom (bps, R2 - A)", 1e4),
        ("cvar95", "Delta Net CVaR95 (bps, R2 - A)", 1e4),
        ("mean_cost", "Delta Mean Tx Cost (bps, R2 - A)", 1e4),
    ]
    for ax, (mk, title, scale) in zip(axes.ravel(), specs):
        means = [stbl[("best_val", sc, mk)]["delta_mean"] * scale for sc in sc_names]
        err_lo = [(stbl[("best_val", sc, mk)]["delta_mean"] - stbl[("best_val", sc, mk)]["ci95_low"]) * scale for sc in sc_names]
        err_hi = [(stbl[("best_val", sc, mk)]["ci95_high"] - stbl[("best_val", sc, mk)]["delta_mean"]) * scale for sc in sc_names]
        colors = ["#1b9e77" if m < 0 else "#d95f02" for m in means]
        x = np.arange(len(sc_names))
        ax.bar(x, means, yerr=[err_lo, err_hi], capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        if mk == "hedged_oce_y_nom":
            ax.axhline(1.0, color="#d7191c", linestyle=":", linewidth=1.2, label="eps=+1.00 bp")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(short_sc)
        ax.set_ylabel("R2_DECAY - A_DECAY (bps)")
        ax.set_title(title)
    fig.tight_layout()
    f2 = plots_dir / "fig2_cost10_6scenarios_best_val.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    # Fig 3: Mechanism breakdown across scenarios: Turnover, Immobility (<1e-5), Active trade size, Max |delta|
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), dpi=150)
    m_specs = [
        ("turnover", "Total Turnover |a| sum"),
        ("immobility_rate", "Immobility Rate (|a| < 1e-5)"),
        ("trade_size_active_pooled", "Mean Trade Size When Active (Pooled)"),
        ("max_abs_delta", "Max |delta|"),
    ]
    x = np.arange(len(sc_names))
    w = 0.36
    for ax, (mk, title) in zip(axes.ravel(), m_specs):
        a_m = [stbl[("best_val", sc, mk)]["A_mean"] for sc in sc_names]
        r2_m = [stbl[("best_val", sc, mk)]["R2_mean"] for sc in sc_names]
        ax.bar(x - w / 2, a_m, width=w, color="#7570b3", label="A_DECAY (10 bps)", edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, r2_m, width=w, color="#1b9e77", label="R2_DECAY (10 bps)", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(short_sc)
        ax.set_title(title)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    f3 = plots_dir / "fig3_cost10_mechanism_gross_net_immobility.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)

    # Fig 4: Sensitivity best_val vs ep200 + Leave-One-Out mean range
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    axes[0].scatter(d_he * 1e4, d_oce * 1e4, c="#1b9e77", edgecolor="black", s=55, label="best_val (primary)")
    axes[0].scatter(d_he_e * 1e4, d_oce_e * 1e4, marker="^", c="#d95f02", edgecolor="black", s=50, alpha=0.8, label="ep200 (sensitivity)")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].axhline(1.0, color="#d7191c", linestyle="--", linewidth=1.2, label="H1b boundary (+1.00 bp)")
    axes[0].set_xlabel("Delta HE_030 (bps)")
    axes[0].set_ylabel("Delta OCE_020 (bps)")
    axes[0].set_title("Trade-Off Scatter (best_val vs ep200)")
    axes[0].legend(fontsize=8.5)

    loo_oce = np.array([np.mean(np.delete(d_oce, i)) * 1e4 for i in range(len(d_oce))])
    axes[1].plot(np.arange(len(seeds)), loo_oce, "o-", color="#2b83ba", markersize=5)
    axes[1].axhline(1.0, color="#d7191c", linestyle="--", linewidth=1.2, label="eps = +1.00 bp")
    axes[1].axhline(m_oce, color="black", linestyle=":", linewidth=1.1, label=f"Full N=24 mean ({m_oce:+.2f} bps)")
    axes[1].set_xticks(np.arange(len(seeds)))
    axes[1].set_xticklabels([str(s)[-2:] for s in seeds], fontsize=8)
    axes[1].set_xlabel("Omitted Seed [50xx]")
    axes[1].set_ylabel("LOO Mean Delta OCE_020 (bps)")
    axes[1].set_title("Leave-One-Out Sensitivity on H1b Mean")
    axes[1].legend(fontsize=8.5)
    fig.tight_layout()
    f4 = plots_dir / "fig4_cost10_best_val_vs_ep200_and_loo.png"
    fig.savefig(f4, dpi=150)
    plt.close(fig)

    return [str(p) for p in (f1, f2, f3, f4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(COST_CONF_DIR / "results" / "confirmatory_n24"))
    parser.add_argument("--plots-dir", type=str, default=str(COST_CONF_DIR / "plots"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = Path(args.plots_dir)

    summary = json.loads((out_dir / "COST_CONFIRMATORY_STATS_SUMMARY.json").read_text(encoding="utf-8"))
    eval_rows = load_csv(out_dir / "RESULTS_COST_CONFIRMATORY_FULL_PRECISION.csv")
    paired_rows = load_csv(out_dir / "PAIRED_COST_CONFIRMATORY.csv")

    recomputed = analyze_cost_confirmatory_pairs(
        load_paired_csvs([out_dir / "PAIRED_COST_CONFIRMATORY.csv"]),
        planned_seeds=COST_LOCKED_COHORT_SEEDS,
    )
    assert recomputed["verdict"] == summary["stats_summary"]["verdict"]
    assert abs(recomputed["p_iut_max"] - summary["stats_summary"]["p_iut_max"]) < 1e-12

    hash_audit = verify_dataset_hash_isolation(summary)
    sha_parity = verify_source_sha_parity(summary, COST_CONF_DIR / "SOURCE_SHA256SUMS.txt")
    tables = extract_seed_table_and_summaries(eval_rows, paired_rows)

    # Save secondary CSV summary
    sec_csv = out_dir / "SECONDARY_COST_SUMMARY_TABLE.csv"
    with open(sec_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tables["scen_table"][0].keys()))
        w.writeheader()
        w.writerows(tables["scen_table"])

    plot_paths = make_plots(out_dir, plots_dir, tables)

    audit_bundle = {
        "verdict": recomputed["verdict"],
        "confirmatory_pass": recomputed["confirmatory_pass"],
        "p_iut_max": recomputed["p_iut_max"],
        "co_primary": recomputed["co_primary"],
        "ledger": recomputed["ledger"],
        "dataset_hash_audit": hash_audit,
        "source_sha_parity": sha_parity,
        "rho_ab_best_val": tables["rho_ab_best_val"],
        "ep200_sensitivity": tables["ep200_sensitivity"],
        "max_cost_accounting_residual": tables["max_cost_accounting_residual"],
        "zero_cost_eval_rows_count": tables["zero_cost_eval_rows_count"],
        "actual_steps_unique": tables["actual_steps_unique"],
        "actual_paths_unique": tables["actual_paths_unique"],
        "extra_mech": tables["extra_mech"],
        "per_seed_rows": tables["per_seed_rows"],
        "plots": plot_paths,
    }
    (out_dir / "POSTRUN_AUDIT_TABLES.json").write_text(json.dumps(audit_bundle, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit_bundle.items() if k != "per_seed_rows"}, indent=2))


if __name__ == "__main__":
    main()
