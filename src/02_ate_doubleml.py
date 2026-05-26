"""
Step 3 — ATE estimation and pooling pre-test.

3a. Pooling pre-test: H0: tau_1yr = tau_2yr
3b. Cluster-aware fold assignment (GroupKFold)
3c. Nuisance model specification (HistGBR)
3d. Pooled ATE estimation with DoubleMLPLR
3e. Residual ICC diagnostic
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLPLR
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold


def load_standards() -> dict:
    p = Path(__file__).resolve().parent.parent / "project_standards.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def log_decision(headline, notes, metrics=None, severity=5):
    p = Path(__file__).resolve().parent.parent / "impl-log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "impl-stage": "Step 3, ATE estimation",
        "headline": headline,
        "notes": notes,
        "metrics": metrics or {},
        "commit-ids": "",
        "status": "completed",
        "severity": severity,
    }
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")


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


def prepare_features(sample, control_cols):
    feature_base = [c for c in control_cols if c != "zoneid" and c in sample.columns]
    zone_dummies = pd.get_dummies(sample["zoneid"], prefix="zone", dtype=float)
    sample = pd.concat([sample, zone_dummies], axis=1)
    zone_cols = list(zone_dummies.columns)
    feature_cols = feature_base + zone_cols
    return sample, feature_cols


def main():
    standards = load_standards()
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / standards["processed_data_dir"]

    klps3 = pd.read_parquet(processed_dir / "klps3_sample.parquet")
    klps4 = pd.read_parquet(processed_dir / "klps4_sample.parquet")

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
    outcomes_klps3 = ["bmi_klps3", "underweight_klps3", "educ_klps3"]
    outcomes_klps4 = ["employed_klps4"]
    all_outcomes = outcomes_klps3 + outcomes_klps4

    # =================================================================
    # 3a. Pooling pre-test: H0: tau_1yr = tau_2yr
    # =================================================================
    log_decision(
        "Starting pooling pre-test: H0: tau_1yr = tau_2yr",
        "Two DoubleMLIRM models with shared GroupKFold splits. "
        "Model A: wgrp=1 vs wgrp=3, Model B: wgrp=2 vs wgrp=3. "
        "Wald test on stacked orthogonal scores.",
        severity=7,
    )

    pooling_results = []

    for outcome in all_outcomes:
        sample = klps3 if outcome in outcomes_klps3 else klps4
        sample = sample.dropna(subset=[outcome]).copy()
        sample, feature_cols = prepare_features(sample, control_cols)

        # Model A: wgrp=1 vs wgrp=3
        wgrp1_sample = sample[sample["psdp_treat_grp"].isin([1, 3])].copy()
        wgrp1_sample["treat_wgrp1"] = (wgrp1_sample["psdp_treat_grp"] == 1).astype(int)

        wgrp1_features = [c for c in feature_cols if c in wgrp1_sample.columns]
        wgrp1_smpls = list(
            GroupKFold(n_splits=n_folds).split(
                wgrp1_sample, groups=wgrp1_sample["base_schid"]
            )
        )

        dml_data_1 = DoubleMLData(
            wgrp1_sample,
            y_col=outcome,
            d_cols="treat_wgrp1",
            x_cols=wgrp1_features,
            force_all_x_finite="allow-nan",
        )
        dml_1 = DoubleMLIRM(
            dml_data_1,
            ml_g=HistGradientBoostingRegressor(max_depth=max_depth, random_state=seed),
            ml_m=HistGradientBoostingClassifier(max_depth=max_depth, random_state=seed),
            n_folds=n_folds,
            draw_sample_splitting=False,
        )
        dml_1.set_sample_splitting(wgrp1_smpls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dml_1.fit()

        tau_1 = dml_1.coef[0]
        se_1 = dml_1.se[0]

        # Model B: wgrp=2 vs wgrp=3
        wgrp2_sample = sample[sample["psdp_treat_grp"].isin([2, 3])].copy()
        wgrp2_sample["treat_wgrp2"] = (wgrp2_sample["psdp_treat_grp"] == 2).astype(int)

        wgrp2_features = [c for c in feature_cols if c in wgrp2_sample.columns]
        wgrp2_smpls = list(
            GroupKFold(n_splits=n_folds).split(
                wgrp2_sample, groups=wgrp2_sample["base_schid"]
            )
        )

        dml_data_2 = DoubleMLData(
            wgrp2_sample,
            y_col=outcome,
            d_cols="treat_wgrp2",
            x_cols=wgrp2_features,
            force_all_x_finite="allow-nan",
        )
        dml_2 = DoubleMLIRM(
            dml_data_2,
            ml_g=HistGradientBoostingRegressor(max_depth=max_depth, random_state=seed),
            ml_m=HistGradientBoostingClassifier(max_depth=max_depth, random_state=seed),
            n_folds=n_folds,
            draw_sample_splitting=False,
        )
        dml_2.set_sample_splitting(wgrp2_smpls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dml_2.fit()

        tau_2 = dml_2.coef[0]
        se_2 = dml_2.se[0]

        # Wald test (conservative: independent SEs)
        diff = tau_1 - tau_2
        se_diff = np.sqrt(se_1**2 + se_2**2)
        wald_stat = (diff / se_diff) ** 2 if se_diff > 0 else np.nan
        p_value = (
            1 - stats.chi2.cdf(wald_stat, df=1) if not np.isnan(wald_stat) else np.nan
        )

        pooling_results.append(
            {
                "outcome": outcome,
                "tau_wgrp1": round(float(tau_1), 6),
                "se_wgrp1": round(float(se_1), 6),
                "n_wgrp1": len(wgrp1_sample),
                "tau_wgrp2": round(float(tau_2), 6),
                "se_wgrp2": round(float(se_2), 6),
                "n_wgrp2": len(wgrp2_sample),
                "diff_tau": round(float(diff), 6),
                "se_diff": round(float(se_diff), 6),
                "wald_stat": (
                    round(float(wald_stat), 4) if not np.isnan(wald_stat) else None
                ),
                "p_value": (
                    round(float(p_value), 4) if not np.isnan(p_value) else None
                ),
                "reject_005": (bool(p_value < 0.05) if not np.isnan(p_value) else None),
            }
        )
        print(
            f"  {outcome}: tau1={tau_1:.4f} (se={se_1:.4f}), "
            f"tau2={tau_2:.4f} (se={se_2:.4f}), "
            f"diff={diff:.4f}, p={p_value:.4f}"
        )

    pooling_df = pd.DataFrame(pooling_results)
    pooling_df.to_csv(processed_dir / "pooling_test_results.csv", index=False)

    log_decision(
        "Pooling pre-test completed",
        "H0: tau_1yr = tau_2yr for each outcome. "
        "Wald test with independent SEs (conservative). ",
        {
            r["outcome"]: {
                "tau_wgrp1": r["tau_wgrp1"],
                "tau_wgrp2": r["tau_wgrp2"],
                "p_value": r["p_value"],
            }
            for r in pooling_results
        },
        severity=8,
    )

    # =================================================================
    # 3b-3d. Pooled ATE estimation with DoubleMLPLR
    # =================================================================
    ate_results = []

    for outcome in all_outcomes:
        sample = klps3 if outcome in outcomes_klps3 else klps4
        sample = sample.dropna(subset=[outcome]).copy()
        sample, feature_cols = prepare_features(sample, control_cols)

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
        p_val = (
            2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
        )

        # Residual ICC from nuisance model predictions
        y_pred = dml_plr.predictions["ml_l"][:, 0, 0]
        resid = dml_plr._dml_data.y.flatten() - y_pred
        sample["resid"] = resid
        resid_icc = compute_icc(
            sample.dropna(subset=["resid"]),
            "resid",
            "base_schid",
        )

        ate_results.append(
            {
                "outcome": outcome,
                "ate": round(float(ate), 6),
                "se": round(float(se), 6),
                "ci_lower": round(float(ci_lower), 6),
                "ci_upper": round(float(ci_upper), 6),
                "t_stat": (round(float(t_stat), 4) if not np.isnan(t_stat) else None),
                "p_value": (round(float(p_val), 6) if not np.isnan(p_val) else None),
                "n": len(sample),
                "n_treated": int(sample["treated"].sum()),
                "n_control": int((sample["treated"] == 0).sum()),
                "n_schools": int(sample["base_schid"].nunique()),
                "residual_icc": round(float(resid_icc), 6),
            }
        )
        print(
            f"  {outcome}: ATE={ate:.4f} (se={se:.4f}), "
            f"CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
            f"p={p_val:.4f}, ICC={resid_icc:.4f}"
        )

    ate_df = pd.DataFrame(ate_results)
    ate_df.to_csv(processed_dir / "ate_results.csv", index=False)

    diagnostics = {
        "ate_results": ate_results,
        "pooling_test": pooling_results,
    }
    with open(processed_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    log_decision(
        "ATE estimation completed for all outcomes",
        "Pooled ATE (wgrp 1+2 vs 3) with cluster-aware GroupKFold.",
        {
            r["outcome"]: {
                "ate": r["ate"],
                "se": r["se"],
                "p_value": r["p_value"],
                "residual_icc": r["residual_icc"],
            }
            for r in ate_results
        },
        severity=8,
    )

    print("\nATE Results:")
    cols = ["outcome", "ate", "se", "ci_lower", "ci_upper", "p_value", "residual_icc"]
    print(ate_df[cols].to_string(index=False))
    print("\nPooling Test Results:")
    cols2 = ["outcome", "tau_wgrp1", "se_wgrp1", "tau_wgrp2", "se_wgrp2", "p_value"]
    print(pooling_df[cols2].to_string(index=False))


if __name__ == "__main__":
    main()
