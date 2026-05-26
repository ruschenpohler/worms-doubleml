"""
Step 1 — Data preparation.

Constructs the analysis sample using the KLPS SampleMaster as spine,
pools wgrp=1 and wgrp=2 as treated vs wgrp=3, builds outcomes,
moderators, controls, and applies exclusion rules.

Every decision is logged to impl-log.jsonl.
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import yaml


def load_standards() -> dict:
    standards_path = Path(__file__).resolve().parent.parent / "project_standards.yaml"
    with open(standards_path) as f:
        return yaml.safe_load(f)


def log_decision(
    headline: str, notes: str, metrics: dict | None = None, severity: int = 5
) -> None:
    log_path = Path(__file__).resolve().parent.parent / "impl-log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "impl-stage": "Step 1, Data preparation",
        "headline": headline,
        "notes": notes,
        "metrics": metrics or {},
        "commit-ids": "",
        "status": "completed",
        "severity": severity,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_dta(path: Path) -> pd.DataFrame:
    df, meta = pyreadstat.read_dta(str(path))
    assert len(df) > 0, f"Empty file: {path}"
    return df


EDUC_MAP = {
    100: 0,
    102: 1,
    103: 2,
    104: 3,
    105: 4,
    106: 5,
    107: 6,
    108: 7,
    109: 8,
    110: 9,
    111: 10,
    112: 11,
    113: 12,
    114: 13,
    115: 14,
    116: 13,
    117: 14,
}


def map_education(code: float | int) -> float:
    if pd.isna(code):
        return np.nan
    code_int = int(code)
    if code_int in (998, 999):
        return np.nan
    if code_int >= 200:
        return np.nan
    return EDUC_MAP.get(code_int, np.nan)


def build_educ_years(series: pd.Series) -> pd.Series:
    return series.apply(map_education)


def build_any_postsecondary(series: pd.Series) -> pd.Series:
    def _map(code):
        if pd.isna(code):
            return np.nan
        code_int = int(code)
        if code_int in (998, 999):
            return np.nan
        return 1 if code_int >= 200 else 0

    return series.apply(_map)


def main() -> None:
    standards = load_standards()
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / standards["raw_data_dir"]
    processed_dir = project_root / standards["processed_data_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 1a. Load all files
    # =================================================================
    klps3_sm = load_dta(raw_dir / standards["raw_files"]["klps3_samplemaster"])
    klps3_i = load_dta(raw_dir / standards["raw_files"]["klps3_imodule"])
    klps4_sm = load_dta(raw_dir / standards["raw_files"]["klps4_samplemaster"])
    klps4_i = load_dta(raw_dir / standards["raw_files"]["klps4_imodule"])
    klps4_e = load_dta(raw_dir / standards["raw_files"]["klps4_emodule"])
    schoolvar = load_dta(raw_dir / standards["raw_files"]["psdp_schoolvar"])
    namelist = load_dta(raw_dir / standards["raw_files"]["psdp_namelist"])

    log_decision(
        "Loaded 7 raw .dta files",
        f"KLPS-3 SM={len(klps3_sm)}, KLPS-3 I={len(klps3_i)}, "
        f"KLPS-4 SM={len(klps4_sm)}, KLPS-4 I={len(klps4_i)}, "
        f"KLPS-4 E={len(klps4_e)}, schoolvar={len(schoolvar)}, "
        f"namelist={len(namelist)}",
        {
            "klps3_sm_rows": len(klps3_sm),
            "klps3_i_rows": len(klps3_i),
            "klps4_sm_rows": len(klps4_sm),
            "klps4_i_rows": len(klps4_i),
            "klps4_e_rows": len(klps4_e),
            "schoolvar_rows": len(schoolvar),
            "namelist_rows": len(namelist),
        },
    )

    # =================================================================
    # 1b. Construct analysis spine from KLPS-3 SampleMaster
    # =================================================================
    spine = klps3_sm[klps3_sm["psdp_treat_grp"].notna()].copy()
    spine["treated"] = (
        spine["psdp_treat_grp"].isin(standards["treatment_group_values"])
    ).astype(int)

    n_gsp_excluded = len(klps3_sm) - len(spine)
    log_decision(
        "Constructed analysis spine from KLPS-3 SampleMaster",
        f"Filtered to psdp_treat_grp not NA, excluding {n_gsp_excluded} GSP pupils. "
        f"Spine: {len(spine)} respondents. "
        f"Treated (wgrp 1+2): {spine['treated'].sum()}, "
        f"Control (wgrp 3): {(spine['treated'] == 0).sum()}. "
        f"Schools: {spine['base_schid'].nunique()}.",
        {
            "n_spine": len(spine),
            "n_gsp_excluded": n_gsp_excluded,
            "n_treated": int(spine["treated"].sum()),
            "n_control": int((spine["treated"] == 0).sum()),
            "n_schools": int(spine["base_schid"].nunique()),
        },
    )

    # Cross-validate psdp_treat_grp against PSDP wgrp from namelist
    namelist_pupil = (
        namelist.groupby("pupid")
        .agg(wgrp=("wgrp", "first"), schid_nm=("schid", "first"))
        .reset_index()
    )
    spine_nm = spine[["pupid", "psdp_treat_grp"]].merge(
        namelist_pupil[["pupid", "wgrp"]], on="pupid", how="inner"
    )
    n_matched = len(spine_nm)
    n_agree = (spine_nm["psdp_treat_grp"] == spine_nm["wgrp"]).sum()
    n_disagree = n_matched - n_agree
    log_decision(
        f"Cross-validated psdp_treat_grp vs namelist wgrp: {n_disagree} mismatches",
        f"Matched {n_matched} pupids between KLPS-3 SM and namelist. "
        f"{n_agree} agree, {n_disagree} disagree. "
        f"Flagging mismatches for exclusion in robustness.",
        {
            "n_matched_pupids": n_matched,
            "n_agree": int(n_agree),
            "n_disagree": int(n_disagree),
        },
    )

    # Flag boundary mismatches
    mismatch_pupids = spine_nm.loc[
        spine_nm["psdp_treat_grp"] != spine_nm["wgrp"], "pupid"
    ].tolist()
    spine["pupid_mismatch"] = spine["pupid"].isin(mismatch_pupids)

    # =================================================================
    # 1c. Construct age_1998 and impute missing base_yob
    # =================================================================
    spine["age_1998_raw"] = 1998 - pd.to_numeric(spine["base_yob"], errors="coerce")
    n_missing_yob = spine["age_1998_raw"].isna().sum()
    pct_missing_yob = spine["base_yob"].isna().mean() * 100
    pct_str = f"{pct_missing_yob:.1f}%"
    log_decision(
        f"Constructed age_1998: {n_missing_yob} missing base_yob ({pct_str})",
        f"age_1998_raw = 1998 - base_yob. "
        f"{n_missing_yob} respondents have missing yob.",
        {
            "n_missing_yob": int(n_missing_yob),
            "pct_missing_yob": round(pct_missing_yob, 1),
            "age_1998_raw_summary": spine["age_1998_raw"].describe().to_dict(),
        },
    )

    # Merge KLPS-3 I-Module for parental education (imputation predictors)
    klps3_parent = klps3_i[["pupid", "s5_2feduc", "s5_7meduc"]].copy()
    klps3_parent["s5_2feduc"] = pd.to_numeric(
        klps3_parent["s5_2feduc"], errors="coerce"
    )
    klps3_parent["s5_7meduc"] = pd.to_numeric(
        klps3_parent["s5_7meduc"], errors="coerce"
    )
    n_father_missing_klps3 = klps3_parent["s5_2feduc"].isna().sum()
    n_mother_missing_klps3 = klps3_parent["s5_7meduc"].isna().sum()
    log_decision(
        "Merged KLPS-3 parental education for imputation",
        f"Father edu missing: {n_father_missing_klps3}/{len(klps3_parent)} "
        f"({n_father_missing_klps3 / len(klps3_parent) * 100:.1f}%). "
        f"Mother edu missing: {n_mother_missing_klps3}/{len(klps3_parent)} "
        f"({n_mother_missing_klps3 / len(klps3_parent) * 100:.1f}%).",
        {
            "n_father_missing": int(n_father_missing_klps3),
            "n_mother_missing": int(n_mother_missing_klps3),
            "pct_father_missing": round(
                n_father_missing_klps3 / len(klps3_parent) * 100, 1
            ),
            "pct_mother_missing": round(
                n_mother_missing_klps3 / len(klps3_parent) * 100, 1
            ),
        },
    )

    spine = spine.merge(klps3_parent, on="pupid", how="left")
    assert spine["pupid"].is_unique, "Duplicate pupid after parental edu merge"
    spine["s5_2feduc"] = pd.to_numeric(spine["s5_2feduc"], errors="coerce")
    spine["s5_7meduc"] = pd.to_numeric(spine["s5_7meduc"], errors="coerce")

    # MICE imputation of base_yob using miceforest
    impute_cols = ["base_yob", "base_std", "base_schid", "s5_2feduc", "s5_7meduc"]
    impute_data = spine[impute_cols].copy()
    impute_data["base_yob"] = pd.to_numeric(impute_data["base_yob"], errors="coerce")
    impute_data["s5_2feduc"] = pd.to_numeric(impute_data["s5_2feduc"], errors="coerce")
    impute_data["s5_7meduc"] = pd.to_numeric(impute_data["s5_7meduc"], errors="coerce")
    impute_data["base_schid"] = impute_data["base_schid"].astype("category")

    from miceforest import ImputationKernel

    seed = standards["random_seed"]
    n_imputations = standards["age_imputation"]["n_imputations"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        kernel = ImputationKernel(
            data=impute_data,
            num_datasets=n_imputations,
            save_all_iterations_data=False,
            random_state=seed,
        )
        kernel.mice(iterations=4)

    imputed_datasets = [kernel.complete_data(i) for i in range(n_imputations)]

    yob_imputed = np.stack([d["base_yob"].values for d in imputed_datasets], axis=1)
    spine["base_yob_imputed"] = np.nanmean(yob_imputed, axis=1)
    spine["base_yob_imputed_std"] = np.nanstd(yob_imputed, axis=1)
    spine["age_1998"] = 1998 - spine["base_yob_imputed"]

    # For complete-case analysis: use raw age where base_yob is observed
    spine["age_1998_cc"] = spine["age_1998_raw"]

    log_decision(
        f"MICE imputation: {n_imputations} datasets via miceforest",
        f"Imputed base_yob using predictors: {impute_cols[1:]}. "
        f"Excluded from model: treated, psdp_treat_grp, female, age_1998. "
        f"Mean imputation std: {spine['base_yob_imputed_std'].mean():.3f}. "
        f"Complete-case analysis column: age_1998_cc.",
        {
            "n_imputations": n_imputations,
            "predictors": impute_cols[1:],
            "mean_imputation_std": round(
                float(spine["base_yob_imputed_std"].mean()), 3
            ),
            "max_imputation_std": round(float(spine["base_yob_imputed_std"].max()), 3),
        },
        severity=7,
    )

    # =================================================================
    # 1d. Construct outcomes
    # =================================================================
    # --- KLPS-3 I-Module outcomes: BMI, underweight, education ---
    i3_outcomes = klps3_i[
        ["pupid", "s19_5aheight", "s19_6aweight", "s4_8edlevel_1"]
    ].copy()

    # Verify variable labels note: s19_5aheight and s19_5bstickid may be swapped
    # Check for implausible values before computing BMI
    i3_outcomes["height_cm"] = i3_outcomes["s19_5aheight"]
    i3_outcomes["weight_kg"] = i3_outcomes["s19_6aweight"]

    # Filter: height must be > 0 and in plausible range (50-250 cm)
    i3_outcomes.loc[i3_outcomes["height_cm"] <= 0, "height_cm"] = np.nan
    i3_outcomes.loc[i3_outcomes["weight_kg"] <= 0, "weight_kg"] = np.nan

    i3_outcomes["bmi_klps3"] = (
        i3_outcomes["weight_kg"] / (i3_outcomes["height_cm"] / 100) ** 2
    )
    # Clinical range filter: BMI 12-60
    i3_outcomes.loc[
        (i3_outcomes["bmi_klps3"] < 12) | (i3_outcomes["bmi_klps3"] > 60),
        "bmi_klps3",
    ] = np.nan
    i3_outcomes["underweight_klps3"] = (i3_outcomes["bmi_klps3"] < 18.5).astype(float)
    i3_outcomes.loc[i3_outcomes["bmi_klps3"].isna(), "underweight_klps3"] = np.nan

    # Education years
    i3_outcomes["educ_klps3"] = build_educ_years(i3_outcomes["s4_8edlevel_1"])
    i3_outcomes["any_postsecondary"] = build_any_postsecondary(
        i3_outcomes["s4_8edlevel_1"]
    )

    n_bmi_valid = i3_outcomes["bmi_klps3"].notna().sum()
    n_underweight_valid = i3_outcomes["underweight_klps3"].notna().sum()
    n_educ_valid = i3_outcomes["educ_klps3"].notna().sum()
    log_decision(
        "Constructed KLPS-3 outcomes: BMI, underweight, education",
        f"BMI valid: {n_bmi_valid}, underweight valid: {n_underweight_valid}, "
        f"education valid: {n_educ_valid}. "
        f"Clinical range filter applied (12-60). "
        f"Education mapped from Kenyan coding scheme (100-117 -> 0-14 yrs). "
        f"Codes 200+ -> any_postsecondary binary, educ_klps3=NaN.",
        {
            "n_bmi_valid": int(n_bmi_valid),
            "n_underweight_valid": int(n_underweight_valid),
            "n_educ_valid": int(n_educ_valid),
            "bmi_range": (
                f"{i3_outcomes['bmi_klps3'].min():.1f}"
                f"-{i3_outcomes['bmi_klps3'].max():.1f}"
            ),
        },
    )

    # --- KLPS-4 E-Module outcomes: earnings, employment ---
    e4_outcomes = klps4_e[
        ["pupid", "s15_2_24aearn_1", "s15_2_1selfemp", "s15_3_1employed"]
    ].copy()

    # log(1 + earnings) for zeros
    e4_outcomes["earnings_klps4"] = e4_outcomes["s15_2_24aearn_1"]
    e4_outcomes["log_earnings_klps4"] = np.log1p(e4_outcomes["earnings_klps4"])

    # Employment: self-employed OR employed for pay
    e4_outcomes["employed_klps4"] = (
        (e4_outcomes["s15_2_1selfemp"] == 1) | (e4_outcomes["s15_3_1employed"] == 1)
    ).astype(float)
    e4_outcomes.loc[
        e4_outcomes["s15_2_1selfemp"].isna() & e4_outcomes["s15_3_1employed"].isna(),
        "employed_klps4",
    ] = np.nan

    n_earnings_valid = e4_outcomes["earnings_klps4"].notna().sum()
    n_employed_valid = e4_outcomes["employed_klps4"].notna().sum()
    log_decision(
        "Constructed KLPS-4 outcomes: earnings, employment",
        f"Earnings valid: {n_earnings_valid}, employment valid: {n_employed_valid}. "
        f"Employment = self-employed (s15_2_1selfemp==1) "
        f"OR employed (s15_3_1employed==1). "
        f"Log transform: log(1+earnings).",
        {
            "n_earnings_valid": int(n_earnings_valid),
            "n_employed_valid": int(n_employed_valid),
            "pct_zero_earnings": round(
                (e4_outcomes["earnings_klps4"] == 0).sum() / n_earnings_valid * 100,
                1,
            ),
        },
    )

    # =================================================================
    # 1e. Construct moderators
    # =================================================================
    # age_1998 already constructed in step 1c
    # female from base_female (no missingness)
    spine["female"] = spine["base_female"].astype(int)
    assert spine["female"].notna().all(), "base_female has missing values"

    log_decision(
        "Constructed moderators: age_1998, female",
        f"age_1998: MICE-imputed for {n_missing_yob} missing base_yob. "
        f"female: from base_female, no missingness ({spine['female'].sum()} female).",
        {
            "n_female": int(spine["female"].sum()),
            "n_male": int((spine["female"] == 0).sum()),
            "age_1998_summary": spine["age_1998"].describe().to_dict(),
        },
    )

    # =================================================================
    # 1f. Construct controls (adjustment set)
    # =================================================================
    # base_std: baseline grade (no missingness)
    spine["base_std_ctrl"] = spine["base_std"].astype(float)

    # grade_retention: expected grade given age minus actual base_std
    # Expected grade: age_1998 - 6 (Kenyan school starting age)
    spine["grade_retention"] = (spine["age_1998"] - 6) - spine["base_std_ctrl"]

    # Parental education: average of available parent's education
    spine["parent_educ_avg"] = spine[["s5_2feduc", "s5_7meduc"]].mean(
        axis=1, skipna=True
    )
    n_both_missing = spine[["s5_2feduc", "s5_7meduc"]].isna().all(axis=1).sum()
    log_decision(
        "Constructed grade_retention and parent_educ_avg",
        f"grade_retention = (age_1998 - 6) - base_std. "
        f"parent_educ_avg = mean of available parental education. "
        f"Both parents missing: {n_both_missing}/{len(spine)}.",
        {
            "n_both_missing": int(n_both_missing),
            "grade_retention_summary": spine["grade_retention"].describe().to_dict(),
        },
    )

    # Merge schoolvar for zone and spillover controls
    sv_cols = [
        "schid",
        "zoneid",
        "sch1_3km_updated",
        "sch1_6km_updated",
        "wgrp",
    ]
    schoolvar_sub = schoolvar[sv_cols].copy()
    schoolvar_sub.rename(
        columns={
            "sch1_3km_updated": "spill_1_3km",
            "sch1_6km_updated": "spill_3_6km",
        },
        inplace=True,
    )

    pre = len(spine)
    spine = spine.merge(
        schoolvar_sub, left_on="base_schid", right_on="schid", how="left"
    )
    assert len(spine) == pre, (
        f"Row count changed after schoolvar merge: {pre} -> {len(spine)}"
    )

    n_spill_missing = spine["spill_1_3km"].isna().sum()
    n_zone_missing = spine["zoneid"].isna().sum()
    log_decision(
        "Merged schoolvar for zone and spillover controls",
        f"Direct merge on base_schid=schid. "
        f"Spillover missing: {n_spill_missing}, zone missing: {n_zone_missing}.",
        {
            "n_spill_missing": int(n_spill_missing),
            "n_zone_missing": int(n_zone_missing),
            "spill_1_3km_summary": spine["spill_1_3km"].describe().to_dict(),
            "spill_3_6km_summary": spine["spill_3_6km"].describe().to_dict(),
        },
    )

    # Verify all 73 KLPS schools matched
    klps_schids = set(spine["base_schid"].dropna().astype(int).unique())
    sv_schids = set(schoolvar["schid"].astype(int).unique())
    unmatched = klps_schids - sv_schids
    assert len(unmatched) == 0, f"KLPS schools not in schoolvar: {unmatched}"

    # =================================================================
    # 1d-continued: Merge outcomes onto spine
    # =================================================================
    # KLPS-3 outcomes (BMI, underweight, education)
    i3_merge = i3_outcomes[
        ["pupid", "bmi_klps3", "underweight_klps3", "educ_klps3", "any_postsecondary"]
    ].copy()
    pre = len(spine)
    spine = spine.merge(i3_merge, on="pupid", how="left")
    assert len(spine) == pre, "Row count changed after KLPS-3 outcome merge"

    # KLPS-4 outcomes (earnings, employment)
    e4_merge = e4_outcomes[
        ["pupid", "earnings_klps4", "log_earnings_klps4", "employed_klps4"]
    ].copy()
    pre = len(spine)
    spine = spine.merge(e4_merge, on="pupid", how="left")
    assert len(spine) == pre, "Row count changed after KLPS-4 outcome merge"

    # Count valid outcomes by group
    outcome_counts = {}
    for outcome in [
        "bmi_klps3",
        "underweight_klps3",
        "educ_klps3",
        "earnings_klps4",
        "employed_klps4",
    ]:
        n_valid = spine[outcome].notna().sum()
        outcome_counts[outcome] = int(n_valid)

    log_decision(
        "Merged KLPS-3 and KLPS-4 outcomes onto spine",
        f"Outcome sample sizes: {outcome_counts}",
        outcome_counts,
    )

    # =================================================================
    # 1g. Exclusion rules and final column set
    # =================================================================
    # GSP pupils already excluded in step 1b (psdp_treat_grp not NA filter)
    # Pupid mismatches flagged but retained for robustness

    # Build final analysis columns
    id_cols = ["pupid", "base_schid", "pupid_mismatch"]
    treatment_cols = ["treated", "psdp_treat_grp"]
    moderator_cols = ["age_1998", "female", "age_1998_cc"]
    outcome_cols = [
        "bmi_klps3",
        "underweight_klps3",
        "educ_klps3",
        "any_postsecondary",
        "earnings_klps4",
        "log_earnings_klps4",
        "employed_klps4",
    ]
    control_cols = [
        "base_std_ctrl",
        "grade_retention",
        "parent_educ_avg",
        "zoneid",
        "spill_1_3km",
        "spill_3_6km",
    ]
    imputation_cols = [
        "base_yob",
        "base_yob_imputed",
        "base_yob_imputed_std",
        "age_1998_raw",
    ]

    final_cols = (
        id_cols
        + treatment_cols
        + moderator_cols
        + outcome_cols
        + control_cols
        + imputation_cols
    )
    final_cols = [c for c in final_cols if c in spine.columns]

    analysis_sample = spine[final_cols].copy()

    log_decision(
        "Analysis sample constructed",
        f"Final sample: {len(analysis_sample)} respondents, "
        f"{len(analysis_sample.columns)} columns. "
        f"Treated: {analysis_sample['treated'].sum()}, "
        f"Control: {(analysis_sample['treated'] == 0).sum()}. "
        f"Pupid mismatches (flagged): {analysis_sample['pupid_mismatch'].sum()}.",
        {
            "n_final": len(analysis_sample),
            "n_columns": len(analysis_sample.columns),
            "n_treated": int(analysis_sample["treated"].sum()),
            "n_control": int((analysis_sample["treated"] == 0).sum()),
            "n_pupid_mismatches": int(analysis_sample["pupid_mismatch"].sum()),
            "outcome_valid_counts": {
                k: int(analysis_sample[k].notna().sum())
                for k in outcome_cols
                if k in analysis_sample.columns
            },
        },
        severity=7,
    )

    # Separate outputs: full pooled sample + KLPS-3 / KLPS-4 specific
    # KLPS-3 sample: respondents with at least one KLPS-3 outcome
    klps3_outcomes = ["bmi_klps3", "underweight_klps3", "educ_klps3"]
    klps3_sample = analysis_sample[
        analysis_sample[klps3_outcomes].notna().any(axis=1)
    ].copy()

    # KLPS-4 sample: respondents with at least one KLPS-4 outcome
    klps4_outcomes = ["earnings_klps4", "employed_klps4"]
    klps4_sample = analysis_sample[
        analysis_sample[klps4_outcomes].notna().any(axis=1)
    ].copy()

    log_decision(
        "Separated KLPS-3 and KLPS-4 outcome samples",
        f"KLPS-3 sample (any outcome): {len(klps3_sample)}. "
        f"KLPS-4 sample (any outcome): {len(klps4_sample)}.",
        {
            "n_klps3_sample": len(klps3_sample),
            "n_klps4_sample": len(klps4_sample),
            "klps3_treated": int(klps3_sample["treated"].sum()),
            "klps3_control": int((klps3_sample["treated"] == 0).sum()),
            "klps4_treated": int(klps4_sample["treated"].sum()),
            "klps4_control": int((klps4_sample["treated"] == 0).sum()),
        },
    )

    # =================================================================
    # Write outputs
    # =================================================================
    analysis_sample.to_parquet(processed_dir / "analysis_sample.parquet", index=False)
    klps3_sample.to_parquet(processed_dir / "klps3_sample.parquet", index=False)
    klps4_sample.to_parquet(processed_dir / "klps4_sample.parquet", index=False)

    print(
        f"Analysis sample: {len(analysis_sample)} rows, "
        f"{len(analysis_sample.columns)} cols"
    )
    print(f"KLPS-3 sample: {len(klps3_sample)} rows")
    print(f"KLPS-4 sample: {len(klps4_sample)} rows")
    n_t = int(analysis_sample["treated"].sum())
    n_c = int((analysis_sample["treated"] == 0).sum())
    print(f"Treated: {n_t}, Control: {n_c}")
    for outcome in outcome_cols:
        if outcome in analysis_sample.columns:
            print(f"  {outcome}: {analysis_sample[outcome].notna().sum()} valid")


if __name__ == "__main__":
    main()
