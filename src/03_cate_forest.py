"""
Step 4 — CATE estimation via causal forest.

4a. Fit CausalForestDML for each outcome
4b. Save CATE estimates indexed by pupid
4c. CATE distribution diagnostics (mean vs ATE, spread)
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


def load_standards() -> dict:
    p = Path(__file__).resolve().parent.parent / "project_standards.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def log_decision(headline, notes, metrics=None, severity=5):
    p = Path(__file__).resolve().parent.parent / "impl-log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "impl-stage": "Step 4, CATE estimation",
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

    ate_results = pd.read_csv(processed_dir / "ate_results.csv")
    ate_lookup = dict(zip(ate_results["outcome"], ate_results["ate"]))

    gkf = GroupKFold(n_splits=n_folds)

    cate_diagnostics = []

    for outcome in all_outcomes:
        sample = klps3 if outcome in outcomes_klps3 else klps4
        sample = sample.dropna(subset=[outcome]).copy()
        sample, feature_cols = prepare_features(sample, control_cols)

        n = len(sample)
        print(f"\n{'=' * 60}")
        print(f"Fitting causal forest for {outcome} (n={n})")
        print(f"{'=' * 60}")

        cf = CausalForestDML(
            model_y=HistGradientBoostingRegressor(
                max_depth=max_depth, random_state=seed
            ),
            model_t=HistGradientBoostingClassifier(
                max_depth=max_depth, random_state=seed
            ),
            n_estimators=4000,
            min_samples_leaf=50,
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

        cate_df = pd.DataFrame(
            {"pupid": sample["pupid"].values, "cate": cate_estimates}
        )
        cate_df.to_parquet(
            processed_dir / f"cate_estimates_{outcome}.parquet", index=False
        )

        ate_val = ate_lookup.get(outcome, np.nan)
        ate_from_cate = cate_estimates.mean()

        age_groups = pd.qcut(sample["age_1998"], 3, labels=["young", "mid", "old"])
        cate_by_tercile = pd.DataFrame(
            {"cate": cate_estimates, "age_tercile": age_groups}
        )
        grp = cate_by_tercile.groupby("age_tercile", observed=False)
        tercile_means = grp["cate"].mean()

        cate_by_sex = pd.DataFrame(
            {"cate": cate_estimates, "female": sample["female"].values}
        )
        sex_means = cate_by_sex.groupby("female")["cate"].mean()

        diag = {
            "outcome": outcome,
            "n": n,
            "ate_step3": round(float(ate_val), 6),
            "mean_cate": round(float(ate_from_cate), 6),
            "std_cate": round(float(cate_estimates.std()), 6),
            "min_cate": round(float(cate_estimates.min()), 6),
            "max_cate": round(float(cate_estimates.max()), 6),
            "cate_young": round(float(tercile_means.get("young", np.nan)), 6),
            "cate_mid": round(float(tercile_means.get("mid", np.nan)), 6),
            "cate_old": round(float(tercile_means.get("old", np.nan)), 6),
            "cate_male": round(float(sex_means.get(0, np.nan)), 6),
            "cate_female": round(float(sex_means.get(1, np.nan)), 6),
        }
        cate_diagnostics.append(diag)

        print(f"  ATE (Step 3):  {ate_val:.4f}")
        print(f"  Mean CATE:     {ate_from_cate:.4f}")
        print(f"  CATE std:      {cate_estimates.std():.4f}")
        print(
            f"  CATE range:    [{cate_estimates.min():.4f}, {cate_estimates.max():.4f}]"
        )
        print("  Age tercile means:")
        print(f"    Young:       {tercile_means.get('young', np.nan):.4f}")
        print(f"    Mid:         {tercile_means.get('mid', np.nan):.4f}")
        print(f"    Old:         {tercile_means.get('old', np.nan):.4f}")
        print("  Sex means:")
        print(f"    Male:        {sex_means.get(0, np.nan):.4f}")
        print(f"    Female:      {sex_means.get(1, np.nan):.4f}")

    cate_diag_df = pd.DataFrame(cate_diagnostics)
    cate_diag_df.to_csv(processed_dir / "cate_diagnostics.csv", index=False)

    log_decision(
        "CATE estimation completed for all outcomes",
        "CausalForestDML with 4000 trees, min_leaf=50, GroupKFold(5), "
        "BLB inference. Moderators: age_1998, female.",
        {
            r["outcome"]: {
                "mean_cate": r["mean_cate"],
                "std_cate": r["std_cate"],
                "ate_step3": r["ate_step3"],
                "cate_young": r["cate_young"],
                "cate_old": r["cate_old"],
            }
            for r in cate_diagnostics
        },
        severity=7,
    )

    print("\n" + "=" * 60)
    print("CATE Diagnostics Summary")
    print("=" * 60)
    cols = [
        "outcome",
        "ate_step3",
        "mean_cate",
        "std_cate",
        "cate_young",
        "cate_old",
        "cate_male",
        "cate_female",
    ]
    print(cate_diag_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
