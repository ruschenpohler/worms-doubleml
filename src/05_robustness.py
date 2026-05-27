"""
Step 5d — Robustness checks.

Check 1: wgrp=1 only (drop partial-treatment group).
  Re-run ATE and BLP for psdp_treat_grp in {1, 3} only.

Check 2: Complete-case age (drop MICE-imputed base_yob).
  Re-run BLP restricting to non-missing base_yob.
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from doubleml import DoubleMLData, DoubleMLPLR
from scipy import stats as sp_stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold


def load_standards():
    p = Path(__file__).resolve().parent.parent / "project_standards.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def log_decision(headline, notes, metrics=None, severity=5):
    p = Path(__file__).resolve().parent.parent / "impl-log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "impl-stage": "Step 5d, robustness",
        "headline": headline,
        "notes": notes,
        "metrics": metrics or {},
        "commit-ids": "",
        "status": "completed",
        "severity": severity,
    }
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")


def prepare_features(sample, control_cols):
    feature_base = [c for c in control_cols if c != "zoneid" and c in sample.columns]
    zone_dummies = pd.get_dummies(sample["zoneid"], prefix="zone", dtype=float)
    sample = pd.concat([sample, zone_dummies], axis=1)
    zone_cols = list(zone_dummies.columns)
    feature_cols = feature_base + zone_cols
    return sample, feature_cols


def compute_icc(df, outcome_col, group_col):
    valid = df[[outcome_col, group_col]].dropna()
    grand_mean = valid[outcome_col].mean()
    ss_between = (
        valid.groupby(group_col)[outcome_col]
        .apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2)
        .sum()
    )
    ss_total = ((valid[outcome_col] - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def run_ate(sample, outcome, feature_cols, n_folds, seed, max_depth):
    gkf = GroupKFold(n_splits=n_folds)
    smpls = list(gkf.split(sample, groups=sample["base_schid"]))

    dml_data = DoubleMLData(
        sample,
        y_col=outcome,
        d_cols="treated",
        x_cols=feature_cols,
        force_all_x_finite="allow-nan",
    )

    dml_plr = DoubleMLPLR(
        dml_data,
        ml_l=HistGradientBoostingRegressor(max_depth=max_depth, random_state=seed),
        ml_m=HistGradientBoostingClassifier(max_depth=max_depth, random_state=seed),
        n_folds=n_folds,
        draw_sample_splitting=False,
    )
    dml_plr.set_sample_splitting(smpls)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dml_plr.fit()

    ate = dml_plr.coef[0]
    se = dml_plr.se[0]
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    t_stat = ate / se if se > 0 else np.nan
    p_val = 2 * (1 - sp_stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan

    y_pred = dml_plr.predictions["ml_l"][:, 0, 0]
    resid = dml_plr._dml_data.y.flatten() - y_pred
    sample_copy = sample.copy()
    sample_copy["resid"] = resid
    resid_icc = compute_icc(sample_copy.dropna(subset=["resid"]), "resid", "base_schid")

    return {
        "outcome": outcome,
        "ate": round(float(ate), 6),
        "se": round(float(se), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": (round(float(p_val), 6) if not np.isnan(p_val) else None),
        "n": len(sample),
        "n_treated": int(sample["treated"].sum()),
        "n_control": int((sample["treated"] == 0).sum()),
        "n_schools": int(sample["base_schid"].nunique()),
        "residual_icc": round(float(resid_icc), 6),
    }


def run_blp(sample, outcome, feature_cols, moderators, n_folds, seed, max_depth):
    age = sample["age_1998"].values
    female = sample["female"].values
    age_x_female = age * female
    school_ids = sample["base_schid"].values

    Y = sample[outcome].values
    T = sample["treated"].values
    W = sample[feature_cols].values
    X_mod = sample[moderators].values
    all_features = np.column_stack([X_mod, W])

    gkf = GroupKFold(n_splits=n_folds)
    Y_hat = np.full(len(Y), np.nan)
    T_hat = np.full(len(T), np.nan)

    for train_idx, test_idx in gkf.split(sample, groups=school_ids):
        model_y = HistGradientBoostingRegressor(max_depth=max_depth, random_state=seed)
        model_t = HistGradientBoostingClassifier(max_depth=max_depth, random_state=seed)
        model_y.fit(all_features[train_idx], Y[train_idx])
        model_t.fit(all_features[train_idx], T[train_idx])
        Y_hat[test_idx] = model_y.predict(all_features[test_idx])
        T_hat[test_idx] = model_t.predict_proba(all_features[test_idx])[:, 1]

    Y_tilde = Y - Y_hat
    T_tilde = T - T_hat

    Z = np.column_stack(
        [
            T_tilde,
            age * T_tilde,
            female * T_tilde,
            age_x_female * T_tilde,
        ]
    )
    col_names = ["intercept", "age_1998", "female", "age_x_female"]

    model = sm.OLS(Y_tilde, Z)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": school_ids})

    blp_result = {
        "outcome": outcome,
        "n": len(sample),
        "n_schools": int(sample["base_schid"].nunique()),
    }
    for i, name in enumerate(col_names):
        blp_result[name] = {
            "coef": round(float(results.params[i]), 6),
            "se": round(float(results.bse[i]), 6),
            "t": round(float(results.tvalues[i]), 4),
            "p": round(float(results.pvalues[i]), 4),
        }

    return blp_result


def main():
    standards = load_standards()
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / standards["processed_data_dir"]

    seed = standards["random_seed"]
    n_folds = standards["n_folds"]
    max_depth = standards["gbt_max_depth"]

    control_cols = [
        "age_1998",
        "female",
        "base_std_ctrl",
        "grade_retention",
        "parent_educ_avg",
        "zoneid",
        "spill_0_3km",
        "spill_3_6km",
    ]
    moderators = ["age_1998", "female"]
    outcomes_klps3 = ["bmi_klps3", "underweight_klps3", "educ_klps3"]
    outcomes_klps4 = ["employed_klps4"]
    all_outcomes = outcomes_klps3 + outcomes_klps4

    # =================================================================
    # Check 1: wgrp=1 only (drop partial treatment group)
    # =================================================================
    print("=" * 60)
    print("SENSITIVITY CHECK 1: wgrp=1 vs wgrp=3 only")
    print("=" * 60)

    klps3 = pd.read_parquet(processed_dir / "klps3_sample.parquet")
    klps4 = pd.read_parquet(processed_dir / "klps4_sample.parquet")

    w1_klps3 = klps3[klps3["psdp_treat_grp"].isin([1, 3])].copy()
    w1_klps3["treated"] = (w1_klps3["psdp_treat_grp"] == 1).astype(int)
    w1_klps4 = klps4[klps4["psdp_treat_grp"].isin([1, 3])].copy()
    w1_klps4["treated"] = (w1_klps4["psdp_treat_grp"] == 1).astype(int)

    ate_w1 = []
    blp_w1 = []

    for outcome in all_outcomes:
        sample = w1_klps3 if outcome in outcomes_klps3 else w1_klps4
        sample = sample.dropna(subset=[outcome]).copy()
        sample, feature_cols = prepare_features(sample, control_cols)

        ate_r = run_ate(sample, outcome, feature_cols, n_folds, seed, max_depth)
        ate_w1.append(ate_r)
        print(
            f"  {outcome}: ATE={ate_r['ate']:.4f} "
            f"(se={ate_r['se']:.4f}), "
            f"p={ate_r['p_value']:.4f}, "
            f"ICC={ate_r['residual_icc']:.4f}"
        )

        blp_r = run_blp(
            sample,
            outcome,
            feature_cols,
            moderators,
            n_folds,
            seed,
            max_depth,
        )
        blp_w1.append(blp_r)
        for coeff in ["age_1998", "female", "age_x_female"]:
            c = blp_r[coeff]
            print(
                f"    {coeff:>15s}: coef={c['coef']:.4f}, "
                f"se={c['se']:.4f}, t={c['t']:.3f}, p={c['p']:.4f}"
            )

    ate_w1_df = pd.DataFrame(ate_w1)
    ate_w1_df.to_csv(processed_dir / "sensitivity_wgrp1_ate.csv", index=False)

    blp_w1_flat = []
    for r in blp_w1:
        for c in ["age_1998", "female", "age_x_female"]:
            blp_w1_flat.append({"outcome": r["outcome"], "coefficient": c, **r[c]})
    blp_w1_df = pd.DataFrame(blp_w1_flat)
    blp_w1_df.to_csv(processed_dir / "sensitivity_wgrp1_blp.csv", index=False)

    log_decision(
        "Sensitivity check 1: wgrp=1 vs wgrp=3 only",
        "Dropped all wgrp=2 (partial treatment) respondents. "
        "ATE and BLP re-estimated with 50 schools.",
        {
            r["outcome"]: {
                "ate": r["ate"],
                "se": r["se"],
                "p": r["p_value"],
            }
            for r in ate_w1
        },
        severity=8,
    )

    # =================================================================
    # Check 2: Complete-case age (drop MICE-imputed base_yob)
    # =================================================================
    print(f"\n{'=' * 60}")
    print("SENSITIVITY CHECK 2: Complete-case age (no imputation)")
    print("=" * 60)

    cc_klps3 = klps3[klps3["age_1998_cc"].notna()].copy()
    cc_klps4 = klps4[klps4["age_1998_cc"].notna()].copy()

    ate_cc = []
    blp_cc = []

    for outcome in all_outcomes:
        sample = cc_klps3 if outcome in outcomes_klps3 else cc_klps4
        sample = sample.dropna(subset=[outcome]).copy()
        sample, feature_cols = prepare_features(sample, control_cols)

        ate_r = run_ate(sample, outcome, feature_cols, n_folds, seed, max_depth)
        ate_cc.append(ate_r)
        print(
            f"  {outcome}: ATE={ate_r['ate']:.4f} "
            f"(se={ate_r['se']:.4f}), "
            f"p={ate_r['p_value']:.4f}, n={ate_r['n']}"
        )

        blp_r = run_blp(
            sample,
            outcome,
            feature_cols,
            moderators,
            n_folds,
            seed,
            max_depth,
        )
        blp_cc.append(blp_r)
        for coeff in ["age_1998", "female", "age_x_female"]:
            c = blp_r[coeff]
            print(
                f"    {coeff:>15s}: coef={c['coef']:.4f}, "
                f"se={c['se']:.4f}, t={c['t']:.3f}, p={c['p']:.4f}"
            )

    ate_cc_df = pd.DataFrame(ate_cc)
    ate_cc_df.to_csv(processed_dir / "sensitivity_complete_case_ate.csv", index=False)

    blp_cc_flat = []
    for r in blp_cc:
        for c in ["age_1998", "female", "age_x_female"]:
            blp_cc_flat.append({"outcome": r["outcome"], "coefficient": c, **r[c]})
    blp_cc_df = pd.DataFrame(blp_cc_flat)
    blp_cc_df.to_csv(processed_dir / "sensitivity_complete_case_blp.csv", index=False)

    log_decision(
        "Sensitivity check 2: complete-case age",
        "Dropped all MICE-imputed base_yob values. "
        "ATE and BLP re-estimated on complete cases.",
        {
            r["outcome"]: {
                "ate": r["ate"],
                "se": r["se"],
                "p": r["p_value"],
            }
            for r in ate_cc
        },
        severity=8,
    )

    # =================================================================
    # Print comparison tables
    # =================================================================
    ate_main = pd.read_csv(processed_dir / "ate_results.csv")
    blp_main = pd.read_csv(processed_dir / "blp_results.csv")

    print(f"\n{'=' * 80}")
    print("COMPARISON: Pooled ATE vs wgrp=1-only ATE")
    print(f"{'=' * 80}")
    for outcome in all_outcomes:
        m = ate_main[ate_main["outcome"] == outcome].iloc[0]
        w = ate_w1_df[ate_w1_df["outcome"] == outcome].iloc[0]
        print(
            f"  {outcome:>20s}: Pooled ATE={m['ate']:.4f} "
            f"(p={m['p_value']:.4f})"
            f"  |  w1-only ATE={w['ate']:.4f} "
            f"(p={w['p_value']:.4f})"
        )

    print(f"\n{'=' * 80}")
    print("COMPARISON: BLP age_1998 coefficients")
    print(f"{'=' * 80}")
    print(
        f"{'Outcome':>20s} | {'Pooled':>20s} | "
        f"{'w1-only':>20s} | {'Complete-case':>20s}"
    )
    print("-" * 88)
    for outcome in all_outcomes:
        m = blp_main[
            (blp_main["outcome"] == outcome) & (blp_main["coefficient"] == "age_1998")
        ].iloc[0]
        w = blp_w1_df[
            (blp_w1_df["outcome"] == outcome) & (blp_w1_df["coefficient"] == "age_1998")
        ].iloc[0]
        c = blp_cc_df[
            (blp_cc_df["outcome"] == outcome) & (blp_cc_df["coefficient"] == "age_1998")
        ].iloc[0]
        print(
            f"  {outcome:>20s} | "
            f"{m['estimate']:.4f} (p={m['p_value']:.4f}) | "
            f"{w['coef']:.4f} (p={w['p']:.4f}) | "
            f"{c['coef']:.4f} (p={c['p']:.4f})"
        )


if __name__ == "__main__":
    main()
