"""
Step 5d (supplement) — Causal forest leaf size sensitivity.

Re-runs CausalForestDML with min_samples_leaf in {20, 50, 100} and
compares CATE distributions and subgroup means. The BLP null is the
primary inference; this checks whether the forest's heterogeneity
patterns are sensitive to the leaf size constraint.
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from econml.dml import CausalForestDML
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
        "impl-stage": "Step 5d supplement, leaf size sensitivity",
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
    moderators = ["age_1998", "female"]
    outcomes_klps3 = ["bmi_klps3", "underweight_klps3", "educ_klps3"]
    outcomes_klps4 = ["employed_klps4"]
    all_outcomes = outcomes_klps3 + outcomes_klps4

    leaf_sizes = [20, 50, 100]

    results = []

    for min_leaf in leaf_sizes:
        print(f"\n{'=' * 70}")
        print(f"min_samples_leaf = {min_leaf}")
        print(f"{'=' * 70}")

        for outcome in all_outcomes:
            sample = klps3 if outcome in outcomes_klps3 else klps4
            sample = sample.dropna(subset=[outcome]).copy()
            sample, feature_cols = prepare_features(sample, control_cols)

            gkf = GroupKFold(n_splits=n_folds)

            cf = CausalForestDML(
                model_y=HistGradientBoostingRegressor(
                    max_depth=max_depth, random_state=seed
                ),
                model_t=HistGradientBoostingClassifier(
                    max_depth=max_depth, random_state=seed
                ),
                n_estimators=4000,
                min_samples_leaf=min_leaf,
                cv=gkf,
                discrete_treatment=True,
                inference="blb",
                random_state=seed,
                allow_missing=True,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf.fit(
                    Y=sample[outcome].values,
                    T=sample["treated"].values,
                    X=sample[moderators].values,
                    W=sample[feature_cols].values,
                    groups=sample["base_schid"].values,
                )

            cate_estimates = cf.effect(sample[moderators].values)

            age_groups = pd.qcut(sample["age_1998"], 3, labels=["young", "mid", "old"])
            cate_df = pd.DataFrame({"cate": cate_estimates, "age_tercile": age_groups})
            tercile_means = cate_df.groupby("age_tercile", observed=False)[
                "cate"
            ].mean()

            sex_means = (
                pd.DataFrame(
                    {
                        "cate": cate_estimates,
                        "female": sample["female"].values,
                    }
                )
                .groupby("female")["cate"]
                .mean()
            )

            row = {
                "outcome": outcome,
                "min_leaf": min_leaf,
                "mean_cate": round(float(cate_estimates.mean()), 6),
                "std_cate": round(float(cate_estimates.std()), 6),
                "min_cate": round(float(cate_estimates.min()), 6),
                "max_cate": round(float(cate_estimates.max()), 6),
                "cate_young": round(float(tercile_means.get("young", np.nan)), 6),
                "cate_mid": round(float(tercile_means.get("mid", np.nan)), 6),
                "cate_old": round(float(tercile_means.get("old", np.nan)), 6),
                "cate_male": round(float(sex_means.get(0, np.nan)), 6),
                "cate_female": round(float(sex_means.get(1, np.nan)), 6),
                "age_spread": round(
                    float(
                        tercile_means.get("young", np.nan)
                        - tercile_means.get("old", np.nan)
                    ),
                    6,
                ),
                "sex_spread": round(
                    float(sex_means.get(1, np.nan) - sex_means.get(0, np.nan)),
                    6,
                ),
            }
            results.append(row)

            print(
                f"  {outcome:>20s}: mean={row['mean_cate']:.4f}, "
                f"std={row['std_cate']:.4f}, "
                f"young={row['cate_young']:.4f}, "
                f"old={row['cate_old']:.4f}, "
                f"male={row['cate_male']:.4f}, "
                f"female={row['cate_female']:.4f}"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(processed_dir / "sensitivity_leaf_size.csv", index=False)

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("LEAF SIZE SENSITIVITY: CATE DISTRIBUTION STATISTICS")
    print(f"{'=' * 80}")
    for outcome in all_outcomes:
        print(f"\n  {outcome}:")
        print(
            f"  {'Leaf':>5s} | {'Mean':>8s} | {'Std':>8s} | "
            f"{'Young':>8s} | {'Old':>8s} | "
            f"{'AgeSprd':>8s} | {'Male':>8s} | "
            f"{'Female':>8s} | {'SexSprd':>8s}"
        )
        print("  " + "-" * 85)
        for min_leaf in leaf_sizes:
            r = results_df[
                (results_df["outcome"] == outcome)
                & (results_df["min_leaf"] == min_leaf)
            ].iloc[0]
            print(
                f"  {min_leaf:>5d} | {r['mean_cate']:>8.4f} | "
                f"{r['std_cate']:>8.4f} | "
                f"{r['cate_young']:>8.4f} | {r['cate_old']:>8.4f} | "
                f"{r['age_spread']:>8.4f} | "
                f"{r['cate_male']:>8.4f} | {r['cate_female']:>8.4f} | "
                f"{r['sex_spread']:>8.4f}"
            )

    log_decision(
        "Leaf size sensitivity completed",
        "CausalForestDML with min_samples_leaf in {20, 50, 100}. "
        "CATE distribution statistics compared across leaf sizes.",
        severity=7,
    )


if __name__ == "__main__":
    main()
