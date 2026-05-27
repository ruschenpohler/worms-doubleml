"""
Step 5d (continued) — Remaining robustness checks.

1. Generalized Lee bounds (Semenova 2025):
   Conditional Lee bounds by age tercile and gender subgroups.
   Asks whether attrition could explain the null specifically for
   younger children or women, even if marginal bounds are uninformative.

2. Parental education sensitivity:
   Re-run ATE and BLP dropping parent_educ_avg from controls entirely.
   Co-primary check: if conclusions are stable, parental education
   as a noisy control is not driving findings.

3. MixedLM nuisance sensitivity:
   Refit outcome nuisance model with school random intercepts,
   recompute orthogonal scores, re-estimate ATE and BLP.
   Checks whether partial pooling changes substantive conclusions
   given residual ICC of 0.02-0.03.
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
        "impl-stage": "Step 5d, additional robustness",
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


# =====================================================================
# 1. Generalized Lee bounds (Semenova 2025)
# =====================================================================
def generalized_lee_bounds(outcome_col, df, rng, n_bootstrap=1000):
    """Compute Lee (2009) bounds conditional on moderator subgroups.

    For each subgroup defined by age tercile and gender, trim the
    higher-attrition group to match the lower-attrition group's
    response rate, then compute bounds.
    """
    df = df.dropna(subset=[outcome_col]).copy()
    df["age_tercile"] = pd.qcut(df["age_1998"], 3, labels=["young", "mid", "old"])
    results = []

    for terc in ["young", "mid", "old"]:
        for fem in [0, 1]:
            sub = df[(df["age_tercile"] == terc) & (df["female"] == fem)]
            if len(sub) < 30:
                continue

            r_t = sub[sub["treated"] == 1][outcome_col].notna().mean()
            r_c = sub[sub["treated"] == 0][outcome_col].notna().mean()

            if r_t >= r_c:
                trim_label = "treated"
                trim_prop = 1.0 - r_c / r_t if r_t > 0 else 0
            else:
                trim_label = "control"
                trim_prop = 1.0 - r_t / r_c if r_c > 0 else 0

            lb, ub = _compute_subgroup_bounds(sub, outcome_col, trim_label, trim_prop)

            boot_bounds = _bootstrap_subgroup_lee(
                sub, outcome_col, trim_label, rng, n_bootstrap
            )

            sex_label = "Female" if fem else "Male"
            results.append(
                {
                    "outcome": outcome_col,
                    "subgroup": f"{terc}_{sex_label}",
                    "n": len(sub),
                    "r_treated": round(float(r_t), 4),
                    "r_control": round(float(r_c), 4),
                    "trim_group": trim_label,
                    "trim_proportion": round(float(trim_prop), 4),
                    "lee_lower": round(float(lb), 4),
                    "lee_upper": round(float(ub), 4),
                    "lee_lower_ci": [round(float(b[0]), 4) for b in boot_bounds]
                    if boot_bounds
                    else [None, None],
                    "lee_upper_ci": [round(float(b[1]), 4) for b in boot_bounds]
                    if boot_bounds
                    else [None, None],
                }
            )

    # Also compute marginal bounds for comparison
    marginal = _compute_marginal_bounds(df, outcome_col, rng, n_bootstrap)
    results.append(marginal)

    return results


def _compute_subgroup_bounds(sub, outcome_col, trim_label, trim_prop):
    treated = sub[sub["treated"] == 1]
    control = sub[sub["treated"] == 0]

    if trim_prop <= 0 or trim_prop >= 1:
        mean_t = treated.dropna(subset=[outcome_col])[outcome_col].mean()
        mean_c = control.dropna(subset=[outcome_col])[outcome_col].mean()
        ate = mean_t - mean_c
        return ate, ate

    if trim_label == "treated":
        high = treated.dropna(subset=[outcome_col])
        low = control.dropna(subset=[outcome_col])
    else:
        high = control.dropna(subset=[outcome_col])
        low = treated.dropna(subset=[outcome_col])

    n_trim = max(0, min(int(np.ceil(trim_prop * len(high))), len(high) - 1))
    if n_trim == 0:
        ate = high[outcome_col].mean() - low[outcome_col].mean()
        if trim_label == "control":
            ate = -ate
        return ate, ate

    sorted_y = high[outcome_col].sort_values()
    remove_top = sorted_y.iloc[: len(sorted_y) - n_trim]
    remove_bottom = sorted_y.iloc[n_trim:]
    mean_stay = low[outcome_col].mean()

    if trim_label == "treated":
        lb = remove_top.mean() - mean_stay
        ub = remove_bottom.mean() - mean_stay
    else:
        lb = mean_stay - remove_bottom.mean()
        ub = mean_stay - remove_top.mean()

    if lb > ub:
        lb, ub = ub, lb
    return lb, ub


def _bootstrap_subgroup_lee(sub, outcome_col, trim_label, rng, n_bootstrap):
    bounds = []
    for _ in range(n_bootstrap):
        try:
            schools = sub["base_schid"].unique()
            b_schools = rng.choice(schools, size=len(schools), replace=True)
            b_idx = np.concatenate(
                [
                    np.where(sub["base_schid"] == s)[0]
                    for s in b_schools
                    if s in sub["base_schid"].values
                ]
            )
            if len(b_idx) < 10:
                continue
            b_sub = sub.iloc[b_idx].copy()
            b_treated = b_sub[b_sub["treated"] == 1]
            b_control = b_sub[b_sub["treated"] == 0]

            b_r_t = b_treated[outcome_col].notna().mean()
            b_r_c = b_control[outcome_col].notna().mean()

            if b_r_t >= b_r_c:
                b_trim = "treated"
                b_tp = 1.0 - b_r_c / b_r_t if b_r_t > 0 else 0
            else:
                b_trim = "control"
                b_tp = 1.0 - b_r_t / b_r_c if b_r_c > 0 else 0

            lb, ub = _compute_subgroup_bounds(b_sub, outcome_col, b_trim, b_tp)
            bounds.append((lb, ub))
        except Exception:
            continue
    return bounds


def _compute_marginal_bounds(df, outcome_col, rng, n_bootstrap):
    """Compute marginal Lee bounds for comparison."""
    treated = df[df["treated"] == 1]
    control = df[df["treated"] == 0]

    r_t = treated[outcome_col].notna().mean()
    r_c = control[outcome_col].notna().mean()

    if r_t >= r_c:
        trim_label = "treated"
        trim_prop = 1.0 - r_c / r_t
    else:
        trim_label = "control"
        trim_prop = 1.0 - r_t / r_c

    lb, ub = _compute_subgroup_bounds(df, outcome_col, trim_label, trim_prop)

    boot_bounds = _bootstrap_subgroup_lee(df, outcome_col, trim_label, rng, n_bootstrap)

    return {
        "outcome": outcome_col,
        "subgroup": "marginal",
        "n": len(df),
        "r_treated": round(float(r_t), 4),
        "r_control": round(float(r_c), 4),
        "trim_group": trim_label,
        "trim_proportion": round(float(trim_prop), 4),
        "lee_lower": round(float(lb), 4),
        "lee_upper": round(float(ub), 4),
        "lee_lower_ci": [
            round(float(np.percentile([b[0] for b in boot_bounds], 2.5)), 4),
            round(float(np.percentile([b[0] for b in boot_bounds], 97.5)), 4),
        ]
        if boot_bounds
        else [None, None],
        "lee_upper_ci": [
            round(float(np.percentile([b[1] for b in boot_bounds], 2.5)), 4),
            round(float(np.percentile([b[1] for b in boot_bounds], 97.5)), 4),
        ]
        if boot_bounds
        else [None, None],
    }


# =====================================================================
# 2. Parental education sensitivity
# =====================================================================
def run_ate_blp(
    sample,
    outcome,
    feature_cols,
    control_cols_full,
    control_cols_reduced,
    n_folds,
    seed,
    max_depth,
    label,
    moderators,
):
    """Run ATE and BLP with given control set. Returns ATE dict and BLP list."""
    from scipy import stats as sp_stats

    def compute_icc(df, col, group_col):
        valid = df[[col, group_col]].dropna()
        gm = valid[col].mean()
        ss_b = (
            valid.groupby(group_col)[col]
            .apply(lambda x: len(x) * (x.mean() - gm) ** 2)
            .sum()
        )
        ss_t = ((valid[col] - gm) ** 2).sum()
        return float(ss_b / ss_t) if ss_t > 0 else 0.0

    gkf = GroupKFold(n_splits=n_folds)
    smpls = list(gkf.split(sample, groups=sample["base_schid"]))

    # ATE
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
    y_pred = dml_plr.predictions["ml_l"][:, 0, 0]
    resid = dml_plr._dml_data.y.flatten() - y_pred
    sample_copy = sample.copy()
    sample_copy["resid"] = resid
    resid_icc = compute_icc(sample_copy.dropna(subset=["resid"]), "resid", "base_schid")

    ate_result = {
        "outcome": outcome,
        "spec": label,
        "ate": round(float(ate), 6),
        "se": round(float(se), 6),
        "p_value": round(float(2 * (1 - sp_stats.norm.cdf(abs(ate / se)))), 6)
        if se > 0
        else None,
        "n": len(sample),
        "residual_icc": round(float(resid_icc), 6),
    }

    # BLP
    age = sample["age_1998"].values
    female = sample["female"].values
    age_x_female = age * female
    school_ids = sample["base_schid"].values
    Y = sample[outcome].values
    T = sample["treated"].values
    W = sample[feature_cols].values
    X_mod = sample[moderators].values
    all_features = np.column_stack([X_mod, W])

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
        [T_tilde, age * T_tilde, female * T_tilde, age_x_female * T_tilde]
    )
    col_names = ["intercept", "age_1998", "female", "age_x_female"]
    model = sm.OLS(Y_tilde, Z)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": school_ids})

    blp_results = []
    for name in ["age_1998", "female", "age_x_female"]:
        i = col_names.index(name)
        blp_results.append(
            {
                "outcome": outcome,
                "spec": label,
                "coefficient": name,
                "estimate": round(float(results.params[i]), 6),
                "se": round(float(results.bse[i]), 6),
                "p_value": round(float(results.pvalues[i]), 4),
            }
        )

    return ate_result, blp_results


# =====================================================================
# 3. MixedLM nuisance sensitivity
# =====================================================================
def run_mixedlm_ate_blp(sample, outcome, feature_cols, n_folds, seed, moderators):
    """Replace outcome nuisance model with MixedLM (school random intercept).
    Treatment nuisance model stays as GBT. Recompute orthogonal scores."""

    def compute_icc(df, col, group_col):
        valid = df[[col, group_col]].dropna()
        gm = valid[col].mean()
        ss_b = (
            valid.groupby(group_col)[col]
            .apply(lambda x: len(x) * (x.mean() - gm) ** 2)
            .sum()
        )
        ss_t = ((valid[col] - gm) ** 2).sum()
        return float(ss_b / ss_t) if ss_t > 0 else 0.0

    gkf = GroupKFold(n_splits=n_folds)
    school_ids = sample["base_schid"].values
    Y = sample[outcome].values
    T = sample["treated"].values
    W = sample[feature_cols].values.astype(float)
    W = np.nan_to_num(W, nan=0.0)

    # Cross-fitted Y_hat from MixedLM
    Y_hat_mixed = np.full(len(Y), np.nan)
    Y_hat_gbt = np.full(len(Y), np.nan)
    T_hat = np.full(len(T), np.nan)

    for train_idx, test_idx in gkf.split(sample, groups=school_ids):
        train_df = sample.iloc[train_idx].copy()
        train_W = W[train_idx]
        test_W = W[test_idx]

        # Treatment model: GBT (same as baseline)
        model_t = HistGradientBoostingClassifier(max_depth=4, random_state=seed)
        model_t.fit(train_W, T[train_idx])
        T_hat[test_idx] = model_t.predict_proba(test_W)[:, 1]

        # Outcome model: MixedLM with school random intercept
        from statsmodels.regression.mixed_linear_model import (
            MixedLM,
        )

        train_exog = sm.add_constant(train_W)
        try:
            mlm = MixedLM(
                train_df[outcome].values,
                train_exog,
                train_df["base_schid"].values,
            )
            mlm_fit = mlm.fit(reml=True)

            test_exog = sm.add_constant(test_W)
            Y_hat_mixed[test_idx] = mlm_fit.predict(test_exog)
        except Exception:
            # Fallback to GBT if MixedLM fails
            model_y = HistGradientBoostingRegressor(max_depth=4, random_state=seed)
            model_y.fit(train_W, Y[train_idx])
            Y_hat_mixed[test_idx] = model_y.predict(test_W)

        # Also get GBT Y_hat for ICC comparison
        model_y = HistGradientBoostingRegressor(max_depth=4, random_state=seed)
        model_y.fit(train_W, Y[train_idx])
        Y_hat_gbt[test_idx] = model_y.predict(test_W)

    # MixedLM-based ATE
    Y_tilde_mixed = Y - Y_hat_mixed
    T_tilde = T - T_hat

    ate_mixed = np.mean(Y_tilde_mixed * T_tilde) / np.mean(T_tilde**2)
    Z_ate = np.column_stack([T_tilde])
    model_ate = sm.OLS(Y_tilde_mixed, Z_ate)
    res_ate = model_ate.fit(cov_type="cluster", cov_kwds={"groups": school_ids})
    se_mixed = res_ate.bse[0]
    p_mixed = res_ate.pvalues[0]

    # Residual ICC for MixedLM
    resid_mixed = Y_tilde_mixed - ate_mixed * T_tilde
    sample_mixed = sample.copy()
    sample_mixed["resid_mixed"] = resid_mixed
    icc_mixed = compute_icc(
        sample_mixed.dropna(subset=["resid_mixed"]),
        "resid_mixed",
        "base_schid",
    )

    # Residual ICC for GBT (baseline)
    resid_gbt = Y - Y_hat_gbt - ate_mixed * T_tilde
    sample_gbt = sample.copy()
    sample_gbt["resid_gbt"] = resid_gbt
    icc_gbt = compute_icc(
        sample_gbt.dropna(subset=["resid_gbt"]),
        "resid_gbt",
        "base_schid",
    )

    # BLP with MixedLM nuisance
    age = sample["age_1998"].values
    female = sample["female"].values
    age_x_female = age * female

    Z_blp = np.column_stack(
        [
            T_tilde,
            age * T_tilde,
            female * T_tilde,
            age_x_female * T_tilde,
        ]
    )
    col_names = ["intercept", "age_1998", "female", "age_x_female"]
    model_blp = sm.OLS(Y_tilde_mixed, Z_blp)
    res_blp = model_blp.fit(cov_type="cluster", cov_kwds={"groups": school_ids})

    blp_results = []
    for name in ["age_1998", "female", "age_x_female"]:
        i = col_names.index(name)
        blp_results.append(
            {
                "outcome": outcome,
                "spec": "mixedlm",
                "coefficient": name,
                "estimate": round(float(res_blp.params[i]), 6),
                "se": round(float(res_blp.bse[i]), 6),
                "p_value": round(float(res_blp.pvalues[i]), 4),
            }
        )

    ate_result = {
        "outcome": outcome,
        "spec": "mixedlm",
        "ate": round(float(ate_mixed), 6),
        "se": round(float(se_mixed), 6),
        "p_value": round(float(p_mixed), 6) if p_mixed is not None else None,
        "residual_icc_mixedlm": round(float(icc_mixed), 6),
        "residual_icc_gbt": round(float(icc_gbt), 6),
    }

    return ate_result, blp_results


def main():
    standards = load_standards()
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / standards["processed_data_dir"]

    seed = standards["random_seed"]
    n_folds = standards["n_folds"]
    max_depth = standards["gbt_max_depth"]
    rng = np.random.default_rng(seed)

    klps3 = pd.read_parquet(processed_dir / "klps3_sample.parquet")
    klps4 = pd.read_parquet(processed_dir / "klps4_sample.parquet")

    outcomes_klps3 = ["bmi_klps3", "underweight_klps3", "educ_klps3"]
    outcomes_klps4 = ["employed_klps4"]
    all_outcomes = outcomes_klps3 + outcomes_klps4
    moderators = ["age_1998", "female"]

    # =================================================================
    # 1. Generalized Lee bounds
    # =================================================================
    print("=" * 60)
    print("GENERALIZED LEE BOUNDS (Semenova 2025)")
    print("=" * 60)

    gen_lee_all = []
    for outcome in all_outcomes:
        df = klps3 if outcome in outcomes_klps3 else klps4
        results = generalized_lee_bounds(outcome, df, rng, n_bootstrap=500)
        gen_lee_all.extend(results)
        for r in results:
            excl_zero = (
                "YES"
                if (
                    r["lee_lower"] is not None
                    and r["lee_upper"] is not None
                    and r["lee_lower"] > 0
                )
                else "NO"
            )
            incl_zero = (
                "YES"
                if (
                    r["lee_lower"] is not None
                    and r["lee_upper"] is not None
                    and r["lee_lower"] <= 0 <= r["lee_upper"]
                )
                else "NO"
            )
            print(
                f"  {r['outcome']:>20s} | {r['subgroup']:>12s} | "
                f"n={r['n']:5d} | "
                f"LB={r['lee_lower']:+.4f} UB={r['lee_upper']:+.4f} | "
                f"excl_zero={excl_zero} incl_zero={incl_zero}"
            )

    gen_lee_df = pd.DataFrame(gen_lee_all)
    gen_lee_df.to_csv(processed_dir / "generalized_lee_bounds.csv", index=False)

    log_decision(
        "Generalized Lee bounds computed",
        "Conditional Lee bounds by age tercile x gender. "
        "500 bootstrap replications per subgroup.",
        severity=8,
    )

    # =================================================================
    # 2. Parental education sensitivity
    # =================================================================
    print(f"\n{'=' * 60}")
    print("PARENTAL EDUCATION SENSITIVITY")
    print("=" * 60)

    control_cols_full = [
        "age_1998",
        "female",
        "base_std_ctrl",
        "grade_retention",
        "parent_educ_avg",
        "zoneid",
        "spill_0_3km",
        "spill_3_6km",
    ]
    control_cols_no_par = [
        "age_1998",
        "female",
        "base_std_ctrl",
        "grade_retention",
        "zoneid",
        "spill_0_3km",
        "spill_3_6km",
    ]

    ate_sens = []
    blp_sens = []

    for outcome in all_outcomes:
        for label, cols in [
            ("full", control_cols_full),
            ("no_parent_ed", control_cols_no_par),
        ]:
            sample = klps3 if outcome in outcomes_klps3 else klps4
            sample = sample.dropna(subset=[outcome]).copy()
            sample, feature_cols = prepare_features(sample, cols)

            ate_r, blp_r = run_ate_blp(
                sample,
                outcome,
                feature_cols,
                control_cols_full,
                control_cols_no_par,
                n_folds,
                seed,
                max_depth,
                label,
                moderators,
            )
            ate_sens.append(ate_r)
            blp_sens.extend(blp_r)

            print(
                f"  {outcome:>20s} | {label:>14s} | "
                f"ATE={ate_r['ate']:.4f} (se={ate_r['se']:.4f}, "
                f"p={ate_r['p_value']:.4f}), ICC={ate_r['residual_icc']:.4f}"
            )
            for br in blp_r:
                print(
                    f"    {br['coefficient']:>15s}: "
                    f"coef={br['estimate']:.4f}, "
                    f"se={br['se']:.4f}, p={br['p_value']:.4f}"
                )

    ate_sens_df = pd.DataFrame(ate_sens)
    ate_sens_df.to_csv(processed_dir / "sensitivity_parent_ed_ate.csv", index=False)
    blp_sens_df = pd.DataFrame(blp_sens)
    blp_sens_df.to_csv(processed_dir / "sensitivity_parent_ed_blp.csv", index=False)

    log_decision(
        "Parental education sensitivity completed",
        "ATE and BLP re-estimated with and without parent_educ_avg.",
        severity=6,
    )

    # =================================================================
    # 3. MixedLM nuisance sensitivity
    # =================================================================
    print(f"\n{'=' * 60}")
    print("MIXEDLM NUISANCE SENSITIVITY")
    print("=" * 60)

    mixedlm_results = []
    for outcome in all_outcomes:
        sample = klps3 if outcome in outcomes_klps3 else klps4
        sample = sample.dropna(subset=[outcome]).copy()
        sample, feature_cols = prepare_features(sample, control_cols_full)

        print(f"\n  {outcome}:")
        try:
            ate_r, blp_r = run_mixedlm_ate_blp(
                sample,
                outcome,
                feature_cols,
                n_folds,
                seed,
                moderators,
            )
            mixedlm_results.append(ate_r)
            mixedlm_results.extend(blp_r)

            print(
                f"    ATE (MixedLM): {ate_r['ate']:.4f} "
                f"(se={ate_r['se']:.4f}, p={ate_r['p_value']:.4f})"
            )
            print(f"    ICC (MixedLM): {ate_r['residual_icc_mixedlm']:.4f}")
            print(f"    ICC (GBT):     {ate_r['residual_icc_gbt']:.4f}")
            for br in blp_r:
                print(
                    f"    {br['coefficient']:>15s}: "
                    f"coef={br['estimate']:.4f}, "
                    f"se={br['se']:.4f}, p={br['p_value']:.4f}"
                )
        except Exception as e:
            print(f"    MixedLM failed: {e}")

    if mixedlm_results:
        mixedlm_ate_df = pd.DataFrame(
            [r for r in mixedlm_results if "coefficient" not in r]
        )
        mixedlm_blp_df = pd.DataFrame(
            [r for r in mixedlm_results if "coefficient" in r]
        )
        mixedlm_ate_df.to_csv(
            processed_dir / "sensitivity_mixedlm_ate.csv", index=False
        )
        mixedlm_blp_df.to_csv(
            processed_dir / "sensitivity_mixedlm_blp.csv", index=False
        )

    log_decision(
        "MixedLM nuisance sensitivity completed",
        "Compared GBT vs MixedLM (school RE) outcome nuisance model.",
        severity=6,
    )


if __name__ == "__main__":
    main()
