"""
Step 2 — Exploratory data analysis.

Surfaces anything that would change a design decision before estimation.
Each subsection ends with an explicit implication for Steps 3-5.

Output: data/processed/eda_summary.csv
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_standards() -> dict:
    standards_path = Path(__file__).resolve().parent.parent / "project_standards.yaml"
    with open(standards_path) as f:
        return yaml.safe_load(f)


def log_decision(
    headline: str, notes: str, metrics: dict | None = None, severity: int = 4
) -> None:
    log_path = Path(__file__).resolve().parent.parent / "impl-log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "impl-stage": "Step 2, EDA",
        "headline": headline,
        "notes": notes,
        "metrics": metrics or {},
        "commit-ids": "",
        "status": "completed",
        "severity": severity,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    standards = load_standards()
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / standards["processed_data_dir"]

    df = pd.read_parquet(processed_dir / "analysis_sample.parquet")
    klps3 = pd.read_parquet(processed_dir / "klps3_sample.parquet")
    klps4 = pd.read_parquet(processed_dir / "klps4_sample.parquet")

    summary_rows = []

    # =============================================================
    # 2a. Sample flow
    # =============================================================
    n_spine = len(df)
    n_klps3 = len(klps3)
    n_klps4 = len(klps4)

    by_treat = df.groupby("treated").agg(
        n_spine=("pupid", "count"),
        n_bmi=("bmi_klps3", lambda x: x.notna().sum()),
        n_educ=("educ_klps3", lambda x: x.notna().sum()),
        n_earn=("earnings_klps4", lambda x: x.notna().sum()),
        n_emp=("employed_klps4", lambda x: x.notna().sum()),
    )
    log_decision(
        "Sample flow: PSDP spine to KLPS-3/4 outcome samples",
        f"Spine: {n_spine}, KLPS-3: {n_klps3}, KLPS-4: {n_klps4}. "
        f"By treatment: {by_treat.to_dict()}",
        {
            "n_spine": n_spine,
            "n_klps3": n_klps3,
            "n_klps4": n_klps4,
            "klps3_treated": int(klps3["treated"].sum()),
            "klps3_control": int((klps3["treated"] == 0).sum()),
            "klps4_treated": int(klps4["treated"].sum()),
            "klps4_control": int((klps4["treated"] == 0).sum()),
        },
    )

    for treat_val, label in [(0, "control"), (1, "treated")]:
        subset = df[df["treated"] == treat_val]
        for outcome in [
            "bmi_klps3",
            "underweight_klps3",
            "educ_klps3",
            "earnings_klps4",
            "employed_klps4",
        ]:
            n_valid = int(subset[outcome].notna().sum())
            n_missing = int(subset[outcome].isna().sum())
            attrition_rate = n_missing / len(subset) if len(subset) > 0 else np.nan
            summary_rows.append(
                {
                    "section": "sample_flow",
                    "variable": outcome,
                    "group": label,
                    "n_valid": n_valid,
                    "n_missing": n_missing,
                    "attrition_rate": round(attrition_rate, 4),
                }
            )

    # =============================================================
    # 2b. Balance check
    # =============================================================
    balance_vars = [
        "age_1998",
        "female",
        "base_std_ctrl",
        "grade_retention",
        "parent_educ_avg",
    ]
    for var in balance_vars:
        treated_vals = df.loc[df["treated"] == 1, var].dropna()
        control_vals = df.loc[df["treated"] == 0, var].dropna()
        diff = treated_vals.mean() - control_vals.mean()
        t_stat, p_val = stats.ttest_ind(treated_vals, control_vals)
        summary_rows.append(
            {
                "section": "balance",
                "variable": var,
                "n_valid": len(treated_vals) + len(control_vals),
                "treated_mean": round(treated_vals.mean(), 4),
                "control_mean": round(control_vals.mean(), 4),
                "diff": round(float(diff), 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_val), 4),
            }
        )

    log_decision(
        "Balance check: moderator and control means by treatment",
        f"Checked {len(balance_vars)} variables. "
        "School-clustered SEs not yet computed (cluster-robust t-test "
        "requires statsmodels).",
        {
            var: {
                "treated_mean": round(float(df.loc[df["treated"] == 1, var].mean()), 4),
                "control_mean": round(float(df.loc[df["treated"] == 0, var].mean()), 4),
            }
            for var in balance_vars
        },
    )

    # =============================================================
    # 2c. Outcome distributions
    # =============================================================
    outcome_stats = {}
    for outcome in [
        "bmi_klps3",
        "underweight_klps3",
        "educ_klps3",
        "earnings_klps4",
        "employed_klps4",
    ]:
        vals = df[outcome].dropna()
        outcome_stats[outcome] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "n": len(vals),
        }
        summary_rows.append(
            {
                "section": "outcome_distribution",
                "variable": outcome,
                "n_valid": len(vals),
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
            }
        )

    log_decision(
        "Outcome distributions",
        f"Summary stats: {outcome_stats}",
        outcome_stats,
    )

    # =============================================================
    # 2d. Differential attrition
    # =============================================================
    for outcome in [
        "bmi_klps3",
        "underweight_klps3",
        "educ_klps3",
        "earnings_klps4",
        "employed_klps4",
    ]:
        df[f"missing_{outcome}"] = df[outcome].isna().astype(int)
        treated_missing = df.loc[df["treated"] == 1, f"missing_{outcome}"].mean()
        control_missing = df.loc[df["treated"] == 0, f"missing_{outcome}"].mean()
        summary_rows.append(
            {
                "section": "differential_attrition",
                "variable": outcome,
                "treated_missing_rate": round(float(treated_missing), 4),
                "control_missing_rate": round(float(control_missing), 4),
                "diff": round(float(treated_missing - control_missing), 4),
            }
        )

    log_decision(
        "Differential attrition by outcome and treatment",
        f"Missingness rates computed for {5} outcomes.",
        {
            outcome: {
                "treated_missing": round(
                    float(df.loc[df["treated"] == 1, f"missing_{outcome}"].mean()),
                    4,
                ),
                "control_missing": round(
                    float(df.loc[df["treated"] == 0, f"missing_{outcome}"].mean()),
                    4,
                ),
            }
            for outcome in [
                "bmi_klps3",
                "underweight_klps3",
                "educ_klps3",
                "earnings_klps4",
                "employed_klps4",
            ]
        },
    )

    # =============================================================
    # 2e. School-level ICC for BMI (primary outcome)
    # =============================================================
    bmi_valid = df[["bmi_klps3", "base_schid"]].dropna()
    grand_mean = bmi_valid["bmi_klps3"].mean()
    school_means = bmi_valid.groupby("base_schid")["bmi_klps3"].mean()
    n_per_school = bmi_valid.groupby("base_schid")["bmi_klps3"].count()
    valid_schools = n_per_school[n_per_school >= 2]
    school_means = school_means.loc[valid_schools.index]
    n_per_school = valid_schools

    ss_between = (n_per_school * (school_means - grand_mean) ** 2).sum()
    ss_total = ((bmi_valid["bmi_klps3"] - grand_mean) ** 2).sum()
    icc_bmi = ss_between / ss_total if ss_total > 0 else 0

    summary_rows.append(
        {
            "section": "icc",
            "variable": "bmi_klps3",
            "icc": round(float(icc_bmi), 4),
            "n_schools": int(len(valid_schools)),
            "grand_mean": round(float(grand_mean), 4),
        }
    )

    log_decision(
        f"School-level ICC for BMI: {icc_bmi:.4f}",
        f"ICC based on {len(valid_schools)} schools with >=2 pupils. "
        f"Grand mean BMI: {grand_mean:.2f}. "
        f"This directly motivates cluster-aware cross-fitting in Step 3.",
        {
            "icc_bmi": round(float(icc_bmi), 4),
            "n_schools": int(len(valid_schools)),
        },
    )

    # =============================================================
    # 2f. Attendance structure (documentation only)
    # =============================================================
    namelist_path = (
        project_root
        / standards["raw_data_dir"]
        / standards["raw_files"]["psdp_namelist"]
    )
    namelist, _ = __import__("pyreadstat").read_dta(str(namelist_path))
    attendance_cols = [c for c in namelist.columns if c.startswith("prs")]
    summary_rows.append(
        {
            "section": "attendance",
            "variable": "namelist_attendance_cols",
            "n_valid": len(attendance_cols),
        }
    )

    log_decision(
        f"Attendance structure documented: {len(attendance_cols)} columns in namelist",
        f"Attendance-related columns: {attendance_cols}. "
        "Per impl-plan, attendance is excluded from estimation. "
        "This is a documentation exercise only.",
        {"attendance_cols": attendance_cols[:10]},
    )

    # =============================================================
    # 2g. Parental education stability check
    # =============================================================
    klps4_i_path = (
        project_root
        / standards["raw_data_dir"]
        / standards["raw_files"]["klps4_imodule"]
    )
    klps4_i, _ = __import__("pyreadstat").read_dta(str(klps4_i_path))

    klps3_i_path = (
        project_root
        / standards["raw_data_dir"]
        / standards["raw_files"]["klps3_imodule"]
    )
    klps3_i, _ = __import__("pyreadstat").read_dta(str(klps3_i_path))

    klps3_parent = klps3_i[["pupid", "s5_2feduc", "s5_7meduc"]].copy()
    klps3_parent["s5_2feduc"] = pd.to_numeric(
        klps3_parent["s5_2feduc"], errors="coerce"
    )
    klps3_parent["s5_7meduc"] = pd.to_numeric(
        klps3_parent["s5_7meduc"], errors="coerce"
    )
    klps3_parent.rename(
        columns={"s5_2feduc": "father_klps3", "s5_7meduc": "mother_klps3"},
        inplace=True,
    )

    klps4_parent = klps4_i[["pupid", "s5_2fatheredu", "s5_7motheredu"]].copy()
    klps4_parent["s5_2fatheredu"] = pd.to_numeric(
        klps4_parent["s5_2fatheredu"], errors="coerce"
    )
    klps4_parent["s5_7motheredu"] = pd.to_numeric(
        klps4_parent["s5_7motheredu"], errors="coerce"
    )
    klps4_parent.rename(
        columns={
            "s5_2fatheredu": "father_klps4",
            "s5_7motheredu": "mother_klps4",
        },
        inplace=True,
    )

    merged_parent = klps3_parent.merge(klps4_parent, on="pupid", how="inner")

    stability_rows = []
    for parent_var, k3_col, k4_col in [
        ("father", "father_klps3", "father_klps4"),
        ("mother", "mother_klps3", "mother_klps4"),
    ]:
        both_present = merged_parent[[k3_col, k4_col]].dropna()
        if len(both_present) > 0:
            exact_agree = (both_present[k3_col] == both_present[k4_col]).mean()
            rho, _ = stats.spearmanr(both_present[k3_col], both_present[k4_col])
        else:
            exact_agree = np.nan
            rho = np.nan
        stability_rows.append(
            {
                "parent": parent_var,
                "n_matched": len(both_present),
                "exact_agreement": round(float(exact_agree), 4),
                "spearman_rho": round(float(rho), 4),
            }
        )
        summary_rows.append(
            {
                "section": "parental_edu_stability",
                "variable": f"{parent_var}_education",
                "n_matched": len(both_present),
                "exact_agreement": round(float(exact_agree), 4),
                "spearman_rho": round(float(rho), 4),
            }
        )

    log_decision(
        "Parental education stability check: KLPS-3 vs KLPS-4",
        f"Father: n={stability_rows[0]['n_matched']}, "
        f"exact_agree={stability_rows[0]['exact_agreement']:.3f}, "
        f"rho={stability_rows[0]['spearman_rho']:.3f}. "
        f"Mother: n={stability_rows[1]['n_matched']}, "
        f"exact_agree={stability_rows[1]['exact_agreement']:.3f}, "
        f"rho={stability_rows[1]['spearman_rho']:.3f}.",
        {
            "father_stability": stability_rows[0],
            "mother_stability": stability_rows[1],
        },
    )

    # =============================================================
    # Write output
    # =============================================================
    eda_df = pd.DataFrame(summary_rows)
    eda_df.to_csv(processed_dir / "eda_summary.csv", index=False)
    print(f"EDA summary written: {len(eda_df)} rows")
    print(f"ICC for BMI: {icc_bmi:.4f}")
    print(f"Parental education stability: {stability_rows}")


if __name__ == "__main__":
    main()
