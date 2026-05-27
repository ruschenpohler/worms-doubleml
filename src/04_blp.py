"""
Step 5a-b — BLP estimation and Romano-Wolf stepdown.

5a. BLP via cross-verified DML:
    tau(X) = alpha_0 + alpha_1*age_1998 + alpha_2*female + alpha_3*(age_1998*female)
    Manual cross-fitting (GroupKFold), second-stage cluster-robust OLS.
5b. Romano-Wolf stepdown on school-level cluster bootstrap of the
    second-stage regression (fast; no refitting of nuisance models).
5c. Lee (2009) bounds for each outcome.
5d. Targeting simulation (top-30% allocation vs random).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
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
        "impl-stage": "Step 5, BLP + inference",
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


def cross_fit_nuisance(df, outcome, feature_cols, moderators, n_folds, seed, groups):
    """Manual cross-fitting. Returns Y_tilde, T_tilde, and
    nuisance-model predictions."""
    Y = df[outcome].values
    T = df["treated"].values
    W = df[feature_cols].values
    X_mod = df[moderators].values

    all_features = np.column_stack([X_mod, W])

    gkf = GroupKFold(n_splits=n_folds)
    Y_hat = np.full(len(Y), np.nan)
    T_hat = np.full(len(T), np.nan)

    for train_idx, test_idx in gkf.split(df, groups=groups):
        model_y = HistGradientBoostingRegressor(max_depth=4, random_state=seed)
        model_t = HistGradientBoostingClassifier(max_depth=4, random_state=seed)
        model_y.fit(all_features[train_idx], Y[train_idx])
        model_t.fit(all_features[train_idx], T[train_idx])

        Y_hat[test_idx] = model_y.predict(all_features[test_idx])
        T_hat[test_idx] = model_t.predict_proba(all_features[test_idx])[:, 1]

    Y_tilde = Y - Y_hat
    T_tilde = T - T_hat

    return Y_tilde, T_tilde


# ------------------------------------------------------------------
# 5a. BLP estimation
# ------------------------------------------------------------------
def estimate_blp(outcome, Y_tilde, T_tilde, age, female, age_x_female, school_ids):
    """Second-stage DML regression with cluster-robust SEs.

    Model: Y_tilde = alpha0*T_tilde + alpha1*(age*T_tilde)
                          + alpha2*(female*T_tilde)
                          + alpha3*(age*female*T_tilde) + epsilon
    """
    Z = np.column_stack(
        [
            T_tilde,
            age * T_tilde,
            female * T_tilde,
            age_x_female * T_tilde,
        ]
    )
    col_names = [
        "intercept",
        "age_1998",
        "female",
        "age_x_female",
    ]
    model = sm.OLS(Y_tilde, Z)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": school_ids})

    coefs = results.params
    ses = results.bse
    t_stats = results.tvalues
    p_values = results.pvalues
    ci_lower = results.conf_int()[:, 0]
    ci_upper = results.conf_int()[:, 1]

    return {
        "outcome": outcome,
        "coefficients": {
            name: {
                "coef": float(coefs[i]),
                "se": float(ses[i]),
                "t": float(t_stats[i]),
                "p": float(p_values[i]),
                "ci_lower": float(ci_lower[i]),
                "ci_upper": float(ci_upper[i]),
            }
            for i, name in enumerate(col_names)
        },
        "Z": Z,
        "Y_tilde": Y_tilde,
    }


# ------------------------------------------------------------------
# 5b. Romano-Wolf stepdown
# ------------------------------------------------------------------
def romano_wolf_stepdown(
    blp_results, outcomes_klps3, outcomes_klps4, rng, n_bootstrap=2000
):
    """Cluster bootstrap on second-stage OLS — resamples schools,
    re-estimates BLP, computes stepdown-adjusted p-values."""
    families = {
        "klps3": outcomes_klps3,
        "klps4": outcomes_klps4,
    }
    rw_adjusted = {}
    for family_name, family_outcomes in families.items():
        obs_abs_t = {}

        for outcome in family_outcomes:
            res = blp_results[outcome]
            for coeff_name in ["age_1998", "female", "age_x_female"]:
                obs_abs_t[(outcome, coeff_name)] = abs(
                    res["coefficients"][coeff_name]["t"]
                )

        all_coeff_keys = [
            (o, c)
            for o in family_outcomes
            for c in ["age_1998", "female", "age_x_female"]
        ]

        shared_schools = None
        for outcome in family_outcomes:
            schools = set(np.unique(blp_results[outcome]["school_ids"]))
            if shared_schools is None:
                shared_schools = schools
            else:
                shared_schools = shared_schools | schools
        shared_schools = np.array(sorted(shared_schools))
        n_schools = len(shared_schools)

        max_t_boot = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            boot_schools = rng.choice(shared_schools, size=n_schools, replace=True)

            all_t = []
            for outcome in family_outcomes:
                Z_full = blp_results[outcome]["Z"]
                Y_full = blp_results[outcome]["Y_tilde"]
                school_ids = blp_results[outcome]["school_ids"]

                boot_idx = np.concatenate(
                    [
                        np.where(school_ids == s)[0]
                        for s in boot_schools
                        if s in school_ids
                    ]
                )
                if len(boot_idx) < 4:
                    for _ in range(3):
                        all_t.append(0.0)
                    continue

                Z_b = Z_full[boot_idx]
                Y_b = Y_full[boot_idx]
                school_b = school_ids[boot_idx]

                try:
                    model_b = sm.OLS(Y_b, Z_b)
                    res_b = model_b.fit(
                        cov_type="cluster",
                        cov_kwds={"groups": school_b},
                    )
                    for j in [1, 2, 3]:
                        all_t.append(abs(res_b.tvalues[j]))
                except Exception:
                    for _ in range(3):
                        all_t.append(0.0)

            max_t_boot[b] = max(all_t) if all_t else 0.0

        stepdown_p = {}
        for key in all_coeff_keys:
            obs_t = obs_abs_t[key]
            rw_p = float(np.mean(max_t_boot >= obs_t))
            stepdown_p[f"{key[0]}__{key[1]}"] = rw_p

        rw_adjusted[family_name] = stepdown_p

    return rw_adjusted


# ------------------------------------------------------------------
# 5c. Lee (2009) bounds
# ------------------------------------------------------------------
def lee_bounds(outcome_col, df, n_bootstrap=1000, rng_seed=42):
    """Lee (2009) trimming bounds for the ATE under monotone selection.

    Procedure:
    1. Identify which group has higher response rate (lower attrition).
    2. Trim that group to match the other group's response rate.
    3. Lower bound: trim top of outcome distribution from higher-response group.
    4. Upper bound: trim bottom of outcome distribution from higher-response group.
    The ATE is always computed as treated_mean - control_mean on the
    comparable (trimmed) sample.
    """
    rng = np.random.default_rng(rng_seed)

    treated = df[df["treated"] == 1]
    control = df[df["treated"] == 0]

    r_t = treated[outcome_col].notna().mean()
    r_c = control[outcome_col].notna().mean()

    if r_t >= r_c:
        trim_label = "treated"
        trim_proportion = 1.0 - r_c / r_t
    else:
        trim_label = "control"
        trim_proportion = 1.0 - r_t / r_c

    def _compute_bounds(treated_df, control_df, trim_prop, trim_from):
        """Core bound computation on a given (sub)sample."""
        if trim_from == "treated":
            high_obs = treated_df.dropna(subset=[outcome_col])
            low_obs = control_df.dropna(subset=[outcome_col])
        else:
            high_obs = control_df.dropna(subset=[outcome_col])
            low_obs = treated_df.dropna(subset=[outcome_col])

        n_trim = int(np.ceil(trim_prop * len(high_obs)))
        n_trim = max(0, min(n_trim, len(high_obs) - 1))

        if n_trim == 0:
            mean_t = treated_df.dropna(subset=[outcome_col])[outcome_col].mean()
            mean_c = control_df.dropna(subset=[outcome_col])[outcome_col].mean()
            return mean_t - mean_c, mean_t - mean_c

        sorted_y = high_obs[outcome_col].sort_values()

        remove_top = sorted_y.iloc[: len(sorted_y) - n_trim]
        remove_bottom = sorted_y.iloc[n_trim:]

        mean_stay = low_obs[outcome_col].mean()

        if trim_from == "treated":
            lb = remove_top.mean() - mean_stay
            ub = remove_bottom.mean() - mean_stay
        else:
            lb = mean_stay - remove_bottom.mean()
            ub = mean_stay - remove_top.mean()

        return lb, ub

    lb_ate, ub_ate = _compute_bounds(treated, control, trim_proportion, trim_label)

    if lb_ate > ub_ate:
        lb_ate, ub_ate = ub_ate, lb_ate

    def _bootstrap_lee(seed):
        b_rng = np.random.default_rng(seed)
        schools = df["base_schid"].unique()
        b_schools = b_rng.choice(schools, size=len(schools), replace=True)
        b_idx = np.concatenate([np.where(df["base_schid"] == s)[0] for s in b_schools])
        b_df = df.iloc[b_idx].copy()
        b_treated = b_df[b_df["treated"] == 1]
        b_control = b_df[b_df["treated"] == 0]
        b_r_t = b_treated[outcome_col].notna().mean()
        b_r_c = b_control[outcome_col].notna().mean()

        if b_r_t >= b_r_c:
            b_trim = "treated"
            b_tp = 1.0 - b_r_c / b_r_t if b_r_t > 0 else 0
        else:
            b_trim = "control"
            b_tp = 1.0 - b_r_t / b_r_c if b_r_c > 0 else 0

        try:
            b_lb, b_ub = _compute_bounds(b_treated, b_control, b_tp, b_trim)
            if b_lb > b_ub:
                b_lb, b_ub = b_ub, b_lb
            return b_lb, b_ub
        except Exception:
            return None

    boot_bounds = [_bootstrap_lee(rng.integers(0, 2**31)) for _ in range(n_bootstrap)]
    boot_bounds = [b for b in boot_bounds if b is not None]

    if boot_bounds:
        boot_lb = [b[0] for b in boot_bounds]
        boot_ub = [b[1] for b in boot_bounds]
        lb_ci = (
            round(float(np.percentile(boot_lb, 2.5)), 4),
            round(float(np.percentile(boot_lb, 97.5)), 4),
        )
        ub_ci = (
            round(float(np.percentile(boot_ub, 2.5)), 4),
            round(float(np.percentile(boot_ub, 97.5)), 4),
        )
    else:
        lb_ci = (None, None)
        ub_ci = (None, None)

    return {
        "outcome": outcome_col,
        "r_treated": round(float(r_t), 4),
        "r_control": round(float(r_c), 4),
        "trim_proportion": round(float(trim_proportion), 4),
        "trim_group": trim_label,
        "lee_lower": round(float(lb_ate), 4),
        "lee_upper": round(float(ub_ate), 4),
        "lee_lower_ci": lb_ci,
        "lee_upper_ci": ub_ci,
    }


# ------------------------------------------------------------------
# 5d. Targeting simulation
# ------------------------------------------------------------------
def targeting_simulation(cate_df, outcome_col, rng, n_bootstrap=1000):
    """If we target treatment to the 30% with highest predicted CATE,
    what is the average gain relative to random allocation?"""
    cate = cate_df["cate"].values
    n = len(cate)
    coverage = 0.30
    n_treated = int(np.ceil(coverage * n))

    sorted_idx = np.argsort(cate)[::-1]
    top_30 = cate[sorted_idx[:n_treated]]

    ate_random = cate.mean()
    ate_targeted = top_30.mean()
    gain = ate_targeted - ate_random

    school_ids = cate_df["base_schid"].values
    unique_schools = np.unique(school_ids)
    boot_gains = np.zeros(n_bootstrap)
    b_rng = np.random.default_rng(42)

    for b in range(n_bootstrap):
        b_schools = b_rng.choice(unique_schools, size=len(unique_schools), replace=True)
        b_idx = np.concatenate([np.where(school_ids == s)[0] for s in b_schools])
        b_cate = cate[b_idx]
        b_sorted = np.argsort(b_cate)[::-1]
        b_top = b_cate[b_sorted[:n_treated]]
        boot_gains[b] = b_top.mean() - b_cate.mean()

    ci_lower = float(np.percentile(boot_gains, 2.5))
    ci_upper = float(np.percentile(boot_gains, 97.5))

    return {
        "outcome": outcome_col,
        "ate_random": round(float(ate_random), 4),
        "ate_targeted_top30": round(float(ate_targeted), 4),
        "gain_vs_random": round(float(gain), 4),
        "gain_ci_lower": round(ci_lower, 4),
        "gain_ci_upper": round(ci_upper, 4),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    standards = load_standards()
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / standards["processed_data_dir"]

    klps3 = pd.read_parquet(processed_dir / "klps3_sample.parquet")
    klps4 = pd.read_parquet(processed_dir / "klps4_sample.parquet")

    seed = standards["random_seed"]
    n_folds = standards["n_folds"]

    rng = np.random.default_rng(seed)

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
    # 5a. BLP estimation
    # =================================================================
    blp_results = {}
    blp_table = []

    for outcome in all_outcomes:
        sample = klps3 if outcome in outcomes_klps3 else klps4
        sample = sample.dropna(subset=[outcome]).copy()
        sample, feature_cols = prepare_features(sample, control_cols)

        age = sample["age_1998"].values
        female = sample["female"].values
        age_x_female = age * female
        school_ids = sample["base_schid"].values

        print(f"\n{'=' * 60}")
        print(f"BLP for {outcome} (n={len(sample)})")
        print(f"{'=' * 60}")

        Y_tilde, T_tilde = cross_fit_nuisance(
            sample, outcome, feature_cols, moderators, n_folds, seed, school_ids
        )

        result = estimate_blp(
            outcome, Y_tilde, T_tilde, age, female, age_x_female, school_ids
        )
        result["school_ids"] = school_ids
        result["age"] = age
        result["female"] = female
        result["age_x_female"] = age_x_female

        blp_results[outcome] = result

        for coeff_name, coeff_data in result["coefficients"].items():
            blp_table.append(
                {
                    "outcome": outcome,
                    "coefficient": coeff_name,
                    "estimate": coeff_data["coef"],
                    "se": coeff_data["se"],
                    "t_stat": coeff_data["t"],
                    "p_value": coeff_data["p"],
                    "ci_lower": coeff_data["ci_lower"],
                    "ci_upper": coeff_data["ci_upper"],
                }
            )
            print(
                f"  {coeff_name:>15s}: "
                f"coef={coeff_data['coef']:.4f}, "
                f"se={coeff_data['se']:.4f}, "
                f"t={coeff_data['t']:.3f}, "
                f"p={coeff_data['p']:.4f}"
            )

    blp_df = pd.DataFrame(blp_table)
    blp_df.to_csv(processed_dir / "blp_results.csv", index=False)

    log_decision(
        "BLP estimation completed",
        "Manual cross-fitting with GroupKFold(5), HistGBR(max_depth=4). "
        "Second-stage OLS with cluster-robust SEs.",
        {
            r["outcome"]: {
                c: r["coefficients"][c]["coef"]
                for c in ["age_1998", "female", "age_x_female"]
            }
            for r in blp_results.values()
        },
        severity=7,
    )

    # =================================================================
    # 5b. Romano-Wolf stepdown
    # =================================================================
    print(f"\n{'=' * 60}")
    print("Romano-Wolf stepdown adjustment")
    print(f"{'=' * 60}")

    rw_adjusted = romano_wolf_stepdown(
        blp_results, outcomes_klps3, outcomes_klps4, rng, n_bootstrap=2000
    )

    rw_rows = []
    for family_name, adj_p in rw_adjusted.items():
        for key, p_val in adj_p.items():
            outcome, coeff_name = key.split("__")
            unadj = blp_results[outcome]["coefficients"][coeff_name]["p"]
            rw_rows.append(
                {
                    "family": family_name,
                    "outcome": outcome,
                    "coefficient": coeff_name,
                    "unadjusted_p": unadj,
                    "rw_adjusted_p": p_val,
                }
            )
            print(
                f"  {family_name:>5s} | {outcome:>20s} | "
                f"{coeff_name:>15s} | "
                f"unadj_p={unadj:.4f} | rw_p={p_val:.4f}"
            )

    rw_df = pd.DataFrame(rw_rows)
    rw_df.to_csv(processed_dir / "rw_stepdown.csv", index=False)

    log_decision(
        "Romano-Wolf stepdown completed",
        "2000 cluster bootstrap replications (school level). "
        "Two families: KLPS-3 (9 tests), KLPS-4 (3 tests).",
        severity=7,
    )

    # =================================================================
    # 5c. Lee (2009) bounds
    # =================================================================
    print(f"\n{'=' * 60}")
    print("Lee (2009) bounds for attrition")
    print(f"{'=' * 60}")

    lee_results = []
    for outcome in all_outcomes:
        df = klps3 if outcome in outcomes_klps3 else klps4
        lb = lee_bounds(outcome, df, n_bootstrap=1000, rng_seed=seed)
        lee_results.append(lb)
        print(
            f"  {outcome:>20s}: "
            f"LB={lb['lee_lower']:.4f}, UB={lb['lee_upper']:.4f}, "
            f"trim={lb['trim_proportion']:.3f} from {lb['trim_group']}"
        )

    lee_df = pd.DataFrame(lee_results)
    lee_df.to_csv(processed_dir / "lee_bounds.csv", index=False)

    log_decision(
        "Lee bounds computed",
        "Lee (2009) trimming bounds for selection bias from attrition. "
        "1000 cluster bootstrap replications for CIs.",
        severity=7,
    )

    # =================================================================
    # 5d. Targeting simulation
    # =================================================================
    print(f"\n{'=' * 60}")
    print("Targeting simulation (top-30% vs random)")
    print(f"{'=' * 60}")

    target_results = []
    for outcome in all_outcomes:
        cate_file = processed_dir / f"cate_estimates_{outcome}.parquet"
        cate_df = pd.read_parquet(cate_file)
        sample_src = klps3 if outcome in outcomes_klps3 else klps4
        cate_df = cate_df.merge(
            sample_src[["pupid", "base_schid"]], on="pupid", how="left"
        )
        tr = targeting_simulation(cate_df, outcome, rng, n_bootstrap=1000)
        target_results.append(tr)
        print(
            f"  {outcome:>20s}: "
            f"ATE_random={tr['ate_random']:.4f}, "
            f"ATE_targeted={tr['ate_targeted_top30']:.4f}, "
            f"gain={tr['gain_vs_random']:.4f} "
            f"[{tr['gain_ci_lower']:.4f}, {tr['gain_ci_upper']:.4f}]"
        )

    target_df = pd.DataFrame(target_results)
    target_df.to_csv(processed_dir / "targeting_simulation.csv", index=False)

    log_decision(
        "Targeting simulation completed",
        "Top-30% CATE allocation vs random; 1000 cluster bootstrap reps.",
        severity=5,
    )

    # =================================================================
    # Print summary tables
    # =================================================================
    print(f"\n{'=' * 60}")
    print("BLP Results Summary")
    print(f"{'=' * 60}")
    print(blp_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("Romano-Wolf Adjusted P-values")
    print(f"{'=' * 60}")
    print(rw_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("Lee Bounds Summary")
    print(f"{'=' * 60}")
    print(lee_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("Targeting Simulation Summary")
    print(f"{'=' * 60}")
    print(target_df.to_string(index=False))


if __name__ == "__main__":
    main()
