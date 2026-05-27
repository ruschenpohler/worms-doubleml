"""
Step 6 — Publication-quality figures.

1. Sample flow
2. Balance table with clustered CIs
3. Outcome distributions by treatment group
4. ATE comparison (DoubleML vs naive OLS)
5. CATE by moderator tercile with BLP caveat note
6. BLP coefficient plot with raw conditional means
7. MDE visualization with observed ATE
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats

PALETTE = {"treated": "#2171b5", "control": "#cb181d"}
OUTCOMES_KLPS3 = ["bmi_klps3", "underweight_klps3", "educ_klps3"]
OUTCOMES_KLPS4 = ["employed_klps4"]
ALL_OUTCOMES = OUTCOMES_KLPS3 + OUTCOMES_KLPS4

OUTCOME_LABELS = {
    "bmi_klps3": "BMI (KLPS-3)",
    "underweight_klps3": "Underweight (KLPS-3)",
    "educ_klps3": "Years of education (KLPS-3)",
    "employed_klps4": "Employment (KLPS-4)",
}

COEFF_LABELS = {
    "intercept": "Intercept",
    "age_1998": "Age at treatment",
    "female": "Female",
    "age_x_female": "Age x Female",
}


def load_standards():
    p = Path(__file__).resolve().parent.parent / "project_standards.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def load_data():
    std = load_standards()
    d = Path(std["processed_data_dir"])
    return {
        "klps3": pd.read_parquet(d / "klps3_sample.parquet"),
        "klps4": pd.read_parquet(d / "klps4_sample.parquet"),
        "ate": pd.read_csv(d / "ate_results.csv"),
        "blp": pd.read_csv(d / "blp_results.csv"),
        "rw": pd.read_csv(d / "rw_stepdown.csv"),
        "lee": pd.read_csv(d / "lee_bounds.csv"),
        "cate": {
            o: pd.read_parquet(d / f"cate_estimates_{o}.parquet") for o in ALL_OUTCOMES
        },
        "processed_dir": d,
    }


def cluster_se(df, col, group_col):
    valid = df[[col, group_col]].dropna()
    grand_mean = valid[col].mean()
    cluster_means = valid.groupby(group_col)[col].mean()
    cluster_sizes = valid.groupby(group_col)[col].count()
    ss_between = (cluster_sizes * (cluster_means - grand_mean) ** 2).sum()
    n_clusters = cluster_means.shape[0]
    var_cluster = ss_between / (n_clusters * (n_clusters - 1))
    return np.sqrt(var_cluster)


# ------------------------------------------------------------------
# Figure 1: Sample flow
# ------------------------------------------------------------------
def fig_sample_flow(data, fig_dir):
    klps3 = data["klps3"]
    klps4 = data["klps4"]

    stages = [
        ("PSDP cohort", len(klps3), None),
    ]
    treated_n = int(klps3["treated"].sum())
    control_n = int((klps3["treated"] == 0).sum())
    stages.append(("PSDP cohort", len(klps3), None))

    bmi_n = klps3.dropna(subset=["bmi_klps3"])
    educ_n = klps3.dropna(subset=["educ_klps3"])
    emp_n = klps4.dropna(subset=["employed_klps4"])

    rows = [
        {
            "stage": "PSDP baseline",
            "total": len(klps3),
            "treated": treated_n,
            "control": control_n,
        },
        {
            "stage": "KLPS-3 respondents",
            "total": len(klps3),
            "treated": int(klps3[klps3["treated"] == 1].shape[0]),
            "control": int(klps3[klps3["treated"] == 0].shape[0]),
        },
        {
            "stage": "  BMI observed",
            "total": len(bmi_n),
            "treated": int(bmi_n[bmi_n["treated"] == 1].shape[0]),
            "control": int(bmi_n[bmi_n["treated"] == 0].shape[0]),
        },
        {
            "stage": "  Education observed",
            "total": len(educ_n),
            "treated": int(educ_n[educ_n["treated"] == 1].shape[0]),
            "control": int(educ_n[educ_n["treated"] == 0].shape[0]),
        },
        {
            "stage": "KLPS-4 respondents",
            "total": len(klps4),
            "treated": int(klps4[klps4["treated"] == 1].shape[0]),
            "control": int(klps4[klps4["treated"] == 0].shape[0]),
        },
        {
            "stage": "  Employment observed",
            "total": len(emp_n),
            "treated": int(emp_n[emp_n["treated"] == 1].shape[0]),
            "control": int(emp_n[emp_n["treated"] == 0].shape[0]),
        },
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_positions = list(range(len(rows)))
    bar_height = 0.35

    for i, row in enumerate(rows):
        ax.barh(
            i + bar_height / 2,
            row["treated"],
            height=bar_height,
            color=PALETTE["treated"],
            label="Treated" if i == 0 else "",
        )
        ax.barh(
            i - bar_height / 2,
            row["control"],
            height=bar_height,
            color=PALETTE["control"],
            label="Control" if i == 0 else "",
        )
        ax.text(
            row["treated"] + 30,
            i + bar_height / 2,
            f"{row['treated']:,}",
            va="center",
            fontsize=8,
            color=PALETTE["treated"],
        )
        ax.text(
            row["control"] + 30,
            i - bar_height / 2,
            f"{row['control']:,}",
            va="center",
            fontsize=8,
            color=PALETTE["control"],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["stage"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Respondents")
    ax.legend(loc="lower right")
    ax.set_title("Sample flow by treatment group")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_sample_flow.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 2: Balance table with clustered CIs
# ------------------------------------------------------------------
def fig_balance(data, fig_dir):
    klps3 = data["klps3"]
    balance_vars = [
        "age_1998",
        "female",
        "base_std_ctrl",
        "grade_retention",
        "parent_educ_avg",
        "spill_0_3km",
        "spill_3_6km",
    ]
    var_labels = {
        "age_1998": "Age at treatment",
        "female": "Female",
        "base_std_ctrl": "Baseline grade",
        "grade_retention": "Grade retention",
        "parent_educ_avg": "Parental education",
        "spill_0_3km": "Schools within 3km",
        "spill_3_6km": "Schools 3-6km",
    }

    results = []
    for var in balance_vars:
        t = klps3[klps3["treated"] == 1][var].dropna()
        c = klps3[klps3["treated"] == 0][var].dropna()
        se_t = cluster_se(klps3[klps3["treated"] == 1], var, "base_schid")
        se_c = cluster_se(klps3[klps3["treated"] == 0], var, "base_schid")
        results.append(
            {
                "var": var_labels[var],
                "treated_mean": t.mean(),
                "treated_se": se_t,
                "control_mean": c.mean(),
                "control_se": se_c,
            }
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(results))
    for i, r in enumerate(results):
        ax.errorbar(
            r["treated_mean"],
            i + 0.15,
            xerr=1.96 * r["treated_se"],
            fmt="o",
            color=PALETTE["treated"],
            capsize=3,
            markersize=5,
        )
        ax.errorbar(
            r["control_mean"],
            i - 0.15,
            xerr=1.96 * r["control_se"],
            fmt="o",
            color=PALETTE["control"],
            capsize=3,
            markersize=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r["var"] for r in results])
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Mean (95% CI, school-clustered)")
    ax.set_title("Balance check: moderator and control means")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color=PALETTE["treated"], label="Treated"),
            plt.Line2D([0], [0], marker="o", color=PALETTE["control"], label="Control"),
        ],
        loc="lower right",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_balance.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 3: Outcome distributions by treatment group
# ------------------------------------------------------------------
def fig_outcome_distributions(data, fig_dir):
    klps3 = data["klps3"]
    klps4 = data["klps4"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_map = {
        "bmi_klps3": axes[0, 0],
        "underweight_klps3": axes[0, 1],
        "educ_klps3": axes[1, 0],
        "employed_klps4": axes[1, 1],
    }

    for outcome, ax in ax_map.items():
        df = klps3 if outcome in OUTCOMES_KLPS3 else klps4
        df = df.dropna(subset=[outcome])
        t = df[df["treated"] == 1][outcome]
        c = df[df["treated"] == 0][outcome]

        if outcome == "underweight_klps3" or outcome == "employed_klps4":
            counts_t = t.value_counts().sort_index()
            counts_c = c.value_counts().sort_index()
            x = sorted(set(counts_t.index) | set(counts_c.index))
            w = 0.35
            ax.bar(
                [xi - w / 2 for xi in x],
                [counts_t.get(xi, 0) / len(t) for xi in x],
                width=w,
                color=PALETTE["treated"],
                alpha=0.8,
                label="Treated",
            )
            ax.bar(
                [xi + w / 2 for xi in x],
                [counts_c.get(xi, 0) / len(c) for xi in x],
                width=w,
                color=PALETTE["control"],
                alpha=0.8,
                label="Control",
            )
            ax.set_ylabel("Proportion")
            ax.set_xticks(x)
        else:
            ax.hist(
                t,
                bins=50,
                density=True,
                alpha=0.5,
                color=PALETTE["treated"],
                label="Treated",
            )
            ax.hist(
                c,
                bins=50,
                density=True,
                alpha=0.5,
                color=PALETTE["control"],
                label="Control",
            )
            ax.set_ylabel("Density")

        ax.set_xlabel(OUTCOME_LABELS[outcome])
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Outcome distributions by treatment group", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_outcome_distributions.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 4: ATE comparison (DoubleML vs naive OLS)
# ------------------------------------------------------------------
def fig_ate_comparison(data, fig_dir):
    ate_df = data["ate"]
    klps3 = data["klps3"]
    klps4 = data["klps4"]

    ols_results = []
    for _, row in ate_df.iterrows():
        outcome = row["outcome"]
        df = klps3 if outcome in OUTCOMES_KLPS3 else klps4
        df = df.dropna(subset=[outcome])
        ols = sm.OLS(df[outcome], sm.add_constant(df[["treated"]])).fit(
            cov_type="cluster", cov_kwds={"groups": df["base_schid"]}
        )
        ols_results.append(
            {
                "outcome": outcome,
                "ols_coef": ols.params["treated"],
                "ols_se": ols.bse["treated"],
            }
        )

    ols_df = pd.DataFrame(ols_results)
    merged = ate_df.merge(ols_df, on="outcome")

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(merged))

    for i, row in merged.iterrows():
        ax.errorbar(
            row["ate"],
            i + 0.15,
            xerr=1.96 * row["se"],
            fmt="s",
            color="#2171b5",
            capsize=4,
            markersize=6,
            label="DoubleML" if i == 0 else "",
        )
        ax.errorbar(
            row["ols_coef"],
            i - 0.15,
            xerr=1.96 * row["ols_se"],
            fmt="o",
            color="#999999",
            capsize=4,
            markersize=6,
            label="Naive OLS" if i == 0 else "",
        )

    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([OUTCOME_LABELS[o] for o in merged["outcome"]])
    ax.set_xlabel("Treatment effect estimate (95% CI)")
    ax.set_title("ATE: DoubleML vs. naive OLS")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_ate_comparison.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 5: CATE by moderator tercile and gender, with BLP caveat
# ------------------------------------------------------------------
def fig_cate_tercile(data, fig_dir):
    klps3 = data["klps3"]
    klps4 = data["klps4"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    blp_caveat = (
        "Tercile differences are not statistically\n"
        "distinguishable from zero (BLP p > 0.10\n"
        "for all moderators, all outcomes)."
    )

    for idx, outcome in enumerate(ALL_OUTCOMES):
        df = klps3 if outcome in OUTCOMES_KLPS3 else klps4
        df = df.dropna(subset=[outcome])
        cate_df = data["cate"][outcome]
        merged = cate_df.merge(df[["pupid", "age_1998", "female"]], on="pupid")

        age_terciles = pd.qcut(merged["age_1998"], 3, labels=["Young", "Mid", "Old"])
        merged["age_tercile"] = age_terciles

        # Panel 1: histogram of CATE
        ax_hist = axes[0, idx]
        ax_hist.hist(merged["cate"], bins=40, density=True, alpha=0.7, color="#6baed6")
        ate_val = data["ate"].loc[data["ate"]["outcome"] == outcome, "ate"].values[0]
        ax_hist.axvline(ate_val, color="red", linewidth=1.5, linestyle="--")
        ax_hist.set_xlabel("CATE")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"{OUTCOME_LABELS[outcome]}")

        # Panel 2: CATE by age tercile and gender
        ax_bar = axes[1, idx]
        groups = merged.groupby(["age_tercile", "female"], observed=False)[
            "cate"
        ].mean()
        labels = []
        means = []
        ses = []
        for terc in ["Young", "Mid", "Old"]:
            for fem, sex_lab in [(0, "M"), (1, "F")]:
                key = (terc, fem)
                if key in groups.index:
                    sub = merged[
                        (merged["age_tercile"] == terc) & (merged["female"] == fem)
                    ]
                    means.append(sub["cate"].mean())
                    labels.append(f"{terc}\n{sex_lab}")
                    se_vals = sub["cate"].std() / np.sqrt(len(sub))
                    ses.append(se_vals)

        x = np.arange(len(means))
        colors = ["#2171b5" if "F" in lab else "#cb181d" for lab in labels]
        ax_bar.bar(x, means, color=colors, alpha=0.7, width=0.6)
        ax_bar.errorbar(
            x,
            means,
            yerr=[1.96 * s for s in ses],
            fmt="none",
            color="black",
            capsize=3,
        )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=7)
        ax_bar.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_bar.set_ylabel("Mean CATE")

    fig.text(0.5, 0.01, blp_caveat, ha="center", fontsize=9, style="italic")
    fig.suptitle("CATE distribution and subgroup means", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(fig_dir / "fig5_cate_tercile.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 6: BLP coefficient plot with raw conditional means
# ------------------------------------------------------------------
def fig_blp_coefficients(data, fig_dir):
    blp = data["blp"]
    rw = data["rw"]

    moderators = ["age_1998", "female", "age_x_female"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

    for idx, outcome in enumerate(ALL_OUTCOMES):
        ax = axes[idx]
        sub_blp = blp[blp["outcome"] == outcome]

        for j, mod in enumerate(moderators):
            row = sub_blp[sub_blp["coefficient"] == mod].iloc[0]
            rw_row = rw[(rw["outcome"] == outcome) & (rw["coefficient"] == mod)]
            rw_p = rw_row["rw_adjusted_p"].values[0]

            unadj_lo = row["ci_lower"]
            unadj_hi = row["ci_upper"]

            z_alpha = 1.96
            rw_z = abs(row["estimate"]) / row["se"] if row["se"] > 0 else 0
            rw_ci_half = z_alpha * row["se"] * max(1, rw_z / z_alpha) if rw_z > 0 else 0
            rw_lo = row["estimate"] - rw_ci_half
            rw_hi = row["estimate"] + rw_ci_half

            y_pos = (2 - j) * 2 + 0.2
            ax.plot(
                [unadj_lo, unadj_hi],
                [y_pos, y_pos],
                color="#2171b5",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.plot(row["estimate"], y_pos, "o", color="#2171b5", markersize=5)

            y_pos_rw = (2 - j) * 2 - 0.2
            z_rw = stats.norm.ppf(1 - rw_p / 2) if rw_p < 1 else 0
            if z_rw > 0:
                rw_half = z_rw * row["se"]
                rw_lo = row["estimate"] - rw_half
                rw_hi = row["estimate"] + rw_half
                ax.plot(
                    [rw_lo, rw_hi],
                    [y_pos_rw, y_pos_rw],
                    color="#cb181d",
                    linewidth=2.5,
                    alpha=0.7,
                )
            ax.plot(row["estimate"], y_pos_rw, "s", color="#cb181d", markersize=4)

        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    labels_y = []
    for mod in moderators:
        labels_y.append(COEFF_LABELS[mod])
    labels_y.append("")
    labels_y.append("")

    axes[0].set_yticks(
        [y for j in range(3) for y in [(2 - j) * 2 + 0.2, (2 - j) * 2 - 0.2]]
    )
    axes[0].set_yticklabels(["unadj", "RW"] * 3, fontsize=7)

    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="#2171b5", label="95% CI (unadjusted)"),
            plt.Line2D([0], [0], color="#cb181d", label="95% CI (Romano-Wolf)"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=9,
    )
    fig.suptitle(
        "BLP coefficients: effect heterogeneity by age and gender",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(fig_dir / "fig6_blp_coefficients.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 7: MDE visualization
# ------------------------------------------------------------------
def fig_mde(data, fig_dir):
    klps3 = data["klps3"]
    bmi = klps3.dropna(subset=["bmi_klps3"])
    treated = bmi[bmi["treated"] == 1]["bmi_klps3"]
    control = bmi[bmi["treated"] == 0]["bmi_klps3"]

    sd_bmi = bmi["bmi_klps3"].std()

    # Unbalanced CRT design: K1=48 treated schools, K0=25 control
    school_counts = bmi.groupby(["base_schid", "treated"]).size().reset_index(name="n")
    K1 = school_counts[school_counts["treated"] == 1]["base_schid"].nunique()
    K0 = school_counts[school_counts["treated"] == 0]["base_schid"].nunique()
    m1 = school_counts[school_counts["treated"] == 1]["n"].mean()
    m0 = school_counts[school_counts["treated"] == 0]["n"].mean()

    from scipy.stats import t as t_dist

    df_clusters = K1 + K0 - 2
    t_alpha = t_dist.ppf(0.975, df=df_clusters)
    t_beta = t_dist.ppf(0.80, df=df_clusters)
    z_sum = t_alpha + t_beta

    # MDE under different ICC assumptions
    # Var(tau) = sigma^2 * [ (1+(m1-1)*ICC)/(K1*m1) + (1+(m0-1)*ICC)/(K0*m0) ]
    icc_scenarios = {
        "Observed (0.023)": 0.023,
        "Conservative (0.05)": 0.05,
        "Very conservative (0.08)": 0.08,
    }
    mde_values = {}
    se_values = {}
    for label, icc in icc_scenarios.items():
        se = sd_bmi * np.sqrt(
            (1 + (m1 - 1) * icc) / (K1 * m1) + (1 + (m0 - 1) * icc) / (K0 * m0)
        )
        mde = z_sum * se
        mde_values[label] = mde
        se_values[label] = se

    ate_bmi = data["ate"].loc[data["ate"]["outcome"] == "bmi_klps3", "ate"].values[0]
    se_bmi = data["ate"].loc[data["ate"]["outcome"] == "bmi_klps3", "se"].values[0]
    mde_observed = z_sum * se_bmi

    # Use the conservative ICC=0.05 as the primary MDE arrow
    mde_conservative = mde_values["Conservative (0.05)"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(12, 45, 80)
    ax.hist(
        control,
        bins=bins,
        density=True,
        alpha=0.4,
        color=PALETTE["control"],
        label=f"Control (n={len(control):,}, {K0} schools)",
    )
    ax.hist(
        treated,
        bins=bins,
        density=True,
        alpha=0.4,
        color=PALETTE["treated"],
        label=f"Treated (n={len(treated):,}, {K1} schools)",
    )

    ax.axvline(control.mean(), color=PALETTE["control"], linestyle="--", linewidth=1.5)
    ax.axvline(treated.mean(), color=PALETTE["treated"], linestyle="--", linewidth=1.5)

    control_mean = control.mean()

    # MDE band (shaded) using conservative ICC
    ax.axvspan(
        control_mean - mde_conservative,
        control_mean + mde_conservative,
        alpha=0.15,
        color="#ff7f0e",
        label=f"MDE (ICC=0.05): [{control_mean - mde_conservative:.1f}, "
        f"{control_mean + mde_conservative:.1f}]",
    )

    # ATE marker
    ax.plot(ate_bmi, 0, "v", color="red", markersize=12, zorder=5)
    y_lim = ax.get_ylim()[1]
    ax.text(
        ate_bmi,
        y_lim * 0.03,
        f"ATE = {ate_bmi:.2f}",
        ha="left",
        fontsize=10,
        color="red",
        fontweight="bold",
    )

    ax.set_xlabel("BMI (kg/m$^2$)")
    ax.set_ylabel("Density")
    ax.set_title("BMI distributions with minimum detectable effect and observed ATE")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    stats_lines = [
        f"Unbalanced CRT: K$_1$={K1}, K$_0$={K0} schools",
        f"Observed ATE = {ate_bmi:.2f} (SE = {se_bmi:.3f})",
        "",
        "MDE at 80% power (t-dist, df={}):".format(df_clusters),
    ]
    for label, mde_val in mde_values.items():
        stats_lines.append(
            f"  ICC {label}: {mde_val:.2f} BMI ({mde_val / sd_bmi:.2f} SD)"
        )
    stats_lines.append(f"  Post-adjustment: {mde_observed:.2f} BMI")

    ax.text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_mde_visualization.pdf", dpi=300)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    fig_dir = project_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data()

    print("Figure 1: Sample flow")
    fig_sample_flow(data, fig_dir)

    print("Figure 2: Balance")
    fig_balance(data, fig_dir)

    print("Figure 3: Outcome distributions")
    fig_outcome_distributions(data, fig_dir)

    print("Figure 4: ATE comparison")
    fig_ate_comparison(data, fig_dir)

    print("Figure 5: CATE by tercile")
    fig_cate_tercile(data, fig_dir)

    print("Figure 6: BLP coefficients")
    fig_blp_coefficients(data, fig_dir)

    print("Figure 7: MDE visualization")
    fig_mde(data, fig_dir)

    print(f"All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
