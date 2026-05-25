"""
Step 1 — Data preparation.

Loads raw .dta files, constructs treatment indicator, merges datasets,
creates analysis variables, applies exclusion rules, and writes the
analysis sample to parquet.

Every decision is logged to impl-log.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyreadstat
import yaml


def load_standards() -> dict:
    """Load project_standards.yaml."""
    standards_path = Path(__file__).resolve().parent.parent / "project_standards.yaml"
    with open(standards_path) as f:
        return yaml.safe_load(f)


def log_decision(
    headline: str, notes: str, metrics: dict | None = None, severity: int = 5
) -> None:
    """Append a decision to impl-log.jsonl."""
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
    """Load a .dta file and assert it has rows."""
    df, _ = pyreadstat.read_dta(str(path))
    assert len(df) > 0, f"Empty file: {path}"
    return df


def main() -> None:
    standards = load_standards()
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / standards["raw_data_dir"]
    processed_dir = project_root / standards["processed_data_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- 1a. Load all files ---
    wormed = load_dta(raw_dir / "wormed.dta")
    pupq = load_dta(raw_dir / "pupq.dta")
    namelist = load_dta(raw_dir / "namelist.dta")
    comply = load_dta(raw_dir / "comply.dta")
    schoolvar = load_dta(raw_dir / "schoolvar.dta")

    log_decision(
        "Loaded 5 raw .dta files",
        f"wormed={len(wormed)}, pupq={len(pupq)}, namelist={len(namelist)}, "
        f"comply={len(comply)}, schoolvar={len(schoolvar)}",
        {
            "wormed_rows": len(wormed),
            "pupq_rows": len(pupq),
            "namelist_rows": len(namelist),
            "comply_rows": len(comply),
            "schoolvar_rows": len(schoolvar),
        },
    )

    # --- 1b. Construct treatment indicator ---
    treated_val = standards["treatment_group_value"]
    partial_val = standards["partial_group_value"]
    comparison_val = standards["comparison_group_value"]

    assert set(namelist["wgrp"].dropna().unique()).issubset(
        {treated_val, partial_val, comparison_val}
    ), f"Unexpected wgrp values: {namelist['wgrp'].unique()}"

    # namelist is at pupil-visit level; collapse to pupil-level
    n_visit_rows = len(namelist)
    n_unique_pupils = namelist["pupid"].nunique()
    namelist_pupil = (
        namelist.groupby("pupid")
        .agg(
            wgrp=("wgrp", "first"),
            sex=("sex", "first"),
            schid=("sch98v1", "first"),
            std=("std98v1", "first"),
        )
        .reset_index()
    )
    assert len(namelist_pupil) == n_unique_pupils, (
        f"Pupil count mismatch: {len(namelist_pupil)} vs {n_unique_pupils}"
    )

    n_before_drop = len(namelist_pupil)
    namelist_pupil = namelist_pupil[namelist_pupil["wgrp"] != partial_val].copy()
    namelist_pupil["treated"] = (namelist_pupil["wgrp"] == treated_val).astype(int)
    n_dropped_partial = n_before_drop - len(namelist_pupil)

    log_decision(
        "Collapsed namelist from pupil-visit to pupil-level",
        f"{n_visit_rows} visit rows to {n_unique_pupils} unique pupils. "
        f"Used sch98v1/std98v1 for baseline school/standard.",
        {"visit_rows": n_visit_rows, "unique_pupils": n_unique_pupils},
    )

    log_decision(
        f"Dropped wgrp={partial_val} (partial treatment)",
        f"Defined treated=(wgrp=={treated_val}). "
        f"Dropped {n_dropped_partial} pupils. "
        f"Remaining: treated={namelist_pupil['treated'].sum()}, "
        f"control={(namelist_pupil['treated'] == 0).sum()}",
        {
            "n_dropped_partial": n_dropped_partial,
            "n_remaining": len(namelist_pupil),
            "n_treated": int(namelist_pupil["treated"].sum()),
            "n_control": int((namelist_pupil["treated"] == 0).sum()),
        },
        severity=7,
    )

    # --- 1c. Merge sequence ---
    # Start with pupil-level namelist as base
    df = namelist_pupil[["pupid", "schid", "wgrp", "sex", "treated"]].copy()

    # LEFT JOIN wormed
    pre = len(df)
    df = df.merge(wormed, on="pupid", how="left", suffixes=("", "_wormed"))
    assert df["pupid"].is_unique, "Duplicate pupid after wormed merge"
    log_decision(
        "Merged wormed on pupid",
        f"Pre: {pre}, Post: {len(df)}, Lost: {pre - len(df)}",
        {"pre": pre, "post": len(df), "lost": pre - len(df)},
    )

    # Cross-check wgrp vs wgrp1
    wgrp_match = (df["wgrp"] == df["wgrp1"]).sum()
    wgrp_mismatch = (df["wgrp"] != df["wgrp1"]).sum()
    if wgrp_mismatch > 0:
        log_decision(
            f"wgrp/wgrp1 mismatch: {wgrp_mismatch} rows",
            f"{wgrp_match} matches, {wgrp_mismatch} mismatches "
            f"between namelist.wgrp and wormed.wgrp1",
            {"wgrp_match": int(wgrp_match), "wgrp_mismatch": int(wgrp_mismatch)},
        )

    # LEFT JOIN pupq
    pre = len(df)
    df = df.merge(pupq, on="pupid", how="left", suffixes=("", "_pupq"))
    assert df["pupid"].is_unique, "Duplicate pupid after pupq merge"
    log_decision(
        "Merged pupq on pupid",
        f"Pre: {pre}, Post: {len(df)}, Lost: {pre - len(df)}",
        {"pre": pre, "post": len(df), "lost": pre - len(df)},
    )

    # LEFT JOIN comply (deduplicate: some pupils have multiple records)
    pre = len(df)
    comply_pupil = comply.groupby("pupid").first().reset_index()
    n_comply_dupes = len(comply) - len(comply_pupil)
    if n_comply_dupes > 0:
        log_decision(
            f"Deduplicated comply: {n_comply_dupes} duplicate rows removed",
            f"{len(comply)} comply rows → {len(comply_pupil)} unique pupils",
            {"comply_rows": len(comply), "comply_unique": len(comply_pupil)},
        )
    df = df.merge(comply_pupil, on="pupid", how="left", suffixes=("", "_comply"))
    assert df["pupid"].is_unique, "Duplicate pupid after comply merge"
    log_decision(
        "Merged comply on pupid",
        f"Pre: {pre}, Post: {len(df)}, Lost: {pre - len(df)}",
        {"pre": pre, "post": len(df), "lost": pre - len(df)},
    )

    # LEFT JOIN schoolvar on schid
    pre = len(df)
    df = df.merge(schoolvar, on="schid", how="left", suffixes=("", "_schoolvar"))
    assert df["pupid"].is_unique, "Duplicate pupid after schoolvar merge"
    log_decision(
        "Merged schoolvar on schid",
        f"Pre: {pre}, Post: {len(df)}, Lost: {pre - len(df)}",
        {"pre": pre, "post": len(df), "lost": pre - len(df)},
    )

    # Attrition: pupils in namelist missing from wormed
    n_missing_wormed = df["hw99"].isna().sum()  # proxy for missing from wormed
    attrition_by_treatment = df.groupby("treated")["hw99"].apply(
        lambda x: x.isna().mean()
    )
    log_decision(
        "Attrition check: missing from wormed",
        f"{n_missing_wormed} pupils missing stool exam data. "
        f"Attrition by treatment: {attrition_by_treatment.to_dict()}",
        {
            "n_missing_wormed": int(n_missing_wormed),
            "attrition_treated_0": float(attrition_by_treatment.get(0, 0)),
            "attrition_treated_1": float(attrition_by_treatment.get(1, 0)),
        },
    )

    # --- 1d. Construct analysis variables ---

    # Primary outcome: infect_heavy_99 (any species moderate-to-heavy by WHO)
    who_cols_99 = ["hw99_who", "sm99_who", "al99_who", "tt99_who"]
    for col in who_cols_99:
        assert col in df.columns, f"Missing WHO column: {col}"
    df["infect_heavy_99"] = df[who_cols_99].max(axis=1).astype(float)

    # Secondary outcome: hb
    assert "hb" in df.columns, "Missing hb column"

    # Moderators
    # infect_intensity_98: sum of baseline egg counts
    egg_cols_98 = ["hw98", "sm98", "al98", "tt98"]
    for col in egg_cols_98:
        assert col in df.columns, f"Missing egg count column: {col}"
    df["infect_intensity_98"] = df[egg_cols_98].sum(axis=1)

    # age_98: from age_98_18
    assert "age_98_18" in df.columns, "Missing age_98_18"
    df["age_98"] = df["age_98_18"].astype(float)

    # female: from sex in namelist, cross-check with sex_98_9 in pupq
    assert "sex" in df.columns, "Missing sex column"
    df["female"] = (df["sex"] == 0).astype(int)  # 0=female, 1=male (verify in EDA)
    if "sex_98_9" in df.columns:
        n_sex_discrepant = (df["sex"] != df["sex_98_9"]).sum()
        if n_sex_discrepant > 0:
            log_decision(
                f"Sex discrepancy: {n_sex_discrepant} rows",
                f"namelist.sex vs pupq.sex_98_9 disagree in {n_sex_discrepant} rows. "
                f"Using namelist.sex as primary source.",
                {"n_sex_discrepant": int(n_sex_discrepant)},
            )

    # Baseline symptom controls (1998 only)
    symptom_map = {
        "headache": "headache_98_45",
        "cough": "cough_98_46",
        "stomach_pain": "stomach_98_57",
        "diarrhea": "diarrhea_98_55",
    }
    symptom_control_cols = []
    for name, col in symptom_map.items():
        if col in df.columns:
            df[f"symptom_{name}"] = df[col].astype(float)
            symptom_control_cols.append(f"symptom_{name}")
        else:
            log_decision(
                f"Missing symptom control: {col}",
                f"Expected baseline symptom column {col} not found. Skipping.",
                severity=4,
            )

    # Zone fixed effects
    assert "zoneid" in df.columns, "Missing zoneid"

    # Spillover controls (updated school count variables, all radii)
    spillover_cols = sorted(
        [c for c in df.columns if c.endswith("_updated") and c.startswith("sch")]
    )

    # --- 1e. Exclusion rules ---
    pre = len(df)

    # Drop missing primary outcome
    df = df[df["infect_heavy_99"].notna()].copy()
    n_drop_outcome = pre - len(df)
    log_decision(
        "Dropped pupils missing primary outcome (infect_heavy_99)",
        f"Dropped {n_drop_outcome} pupils",
        {"n_dropped_missing_outcome": n_drop_outcome},
    )

    # Drop pupils missing both gender variables
    pre = len(df)
    has_sex = df["sex"].notna()
    has_sex_98 = (
        df["sex_98_9"].notna()
        if "sex_98_9" in df.columns
        else pd.Series(False, index=df.index)
    )
    df = df[has_sex | has_sex_98].copy()
    n_drop_gender = pre - len(df)
    if n_drop_gender > 0:
        log_decision(
            "Dropped pupils missing both gender variables",
            f"Dropped {n_drop_gender} pupils",
            {"n_dropped_missing_gender": n_drop_gender},
        )

    # --- Build final column set ---
    id_cols = ["pupid", "schid"]
    treatment_cols = ["treated", "wgrp"]
    outcome_cols = ["infect_heavy_99", "hb"]
    moderator_cols = ["infect_intensity_98", "age_98", "female"]
    control_cols = ["zoneid"] + symptom_control_cols + spillover_cols
    geo_cols = ["distlake"] if "distlake" in df.columns else []

    final_cols = (
        id_cols
        + treatment_cols
        + outcome_cols
        + moderator_cols
        + control_cols
        + geo_cols
    )
    final_cols = [c for c in final_cols if c in df.columns]

    analysis_sample = df[final_cols].copy()

    log_decision(
        "Analysis sample constructed",
        f"Final sample: {len(analysis_sample)} pupils, "
        f"{len(analysis_sample.columns)} columns. "
        f"Treated: {analysis_sample['treated'].sum()}, "
        f"Control: {(analysis_sample['treated'] == 0).sum()}",
        {
            "n_final": len(analysis_sample),
            "n_columns": len(analysis_sample.columns),
            "n_treated": int(analysis_sample["treated"].sum()),
            "n_control": int((analysis_sample["treated"] == 0).sum()),
        },
    )

    # --- Output ---
    out_path = processed_dir / "analysis_sample.parquet"
    analysis_sample.to_parquet(out_path, index=False)
    print(f"Analysis sample written to {out_path}")
    print(f"  Rows: {len(analysis_sample)}")
    print(f"  Columns: {len(analysis_sample.columns)}")
    print(f"  Treated: {analysis_sample['treated'].sum()}")
    print(f"  Control: {(analysis_sample['treated'] == 0).sum()}")


if __name__ == "__main__":
    main()
