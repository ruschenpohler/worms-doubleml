"""
Step 6 — Publication-quality figures.

New numbering (matches readme order):
1. Sample flow (bold stage labels, updated text)
2. Balance check (parental education in years, split axis)
3. BLP coefficients with Romano-Wolf CIs (2x2 grid)
4. Outcome distributions by treatment group
5. ATE comparison (DoubleML vs naive OLS)
6. MDE visualization with observed ATE (table notes, no legend clutter)
7. CATE distribution and tercile subgroup means (two separate 2x2 panels)
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
    "bmi_klps3": "BMI",
    "underweight_klps3": "Underweight",
    "educ_klps3": "Education (years)",
    "employed_klps4": "Employment",
}

COEFF_LABELS = {
    "intercept": "Intercept",
    "age_1998": "Age at treatment",
    "female": "Female",
    "age_x_female": "Age x Female",
}

TITLE_MAP = {
    1: "Figure 1: Sample flow by treatment group",
    2: "Figure 2: Covariate balance by treatment group",
    3: "Figure 3: BLP coefficients with Romano-Wolf adjusted confidence intervals",
    4: "Figure 4: Outcome distributions by treatment group",
    5: "Figure 5: ATE estimates under DoubleML and naive OLS",
    6: "Figure 6: BMI distributions with minimum detectable effect and observed ATE",
    7: "Figure 7: CATE distributions and subgroup means by moderator tercile",
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

    bmi_n = klps3.dropna(subset=["bmi_klps3"])
    educ_n = klps3.dropna(subset=["educ_klps3"])
    emp_n = klps4.dropna(subset=["employed_klps4"])

    rows = [
        {
            "stage": "PSDP baseline",
            "total": len(klps3),
            "treated": int(klps3[klps3["treated"] == 1].shape[0]),
            "control": int(klps3[klps3["treated"] == 0].shape[0]),
            "bold": True,
        },
        {
            "stage": "KLPS-3 (2011-2014)",
            "total": len(klps3),
            "treated": int(klps3[klps3["treated"] == 1].shape[0]),
            "control": int(klps3[klps3["treated"] == 0].shape[0]),
            "bold": True,
        },
        {
            "stage": "  Body mass index",
            "total": len(bmi_n),
            "treated": int(bmi_n[bmi_n["treated"] == 1].shape[0]),
            "control": int(bmi_n[bmi_n["treated"] == 0].shape[0]),
            "bold": False,
        },
        {
            "stage": "  Education (in years)",
            "total": len(educ_n),
            "treated": int(educ_n[educ_n["treated"] == 1].shape[0]),
            "control": int(educ_n[educ_n["treated"] == 0].shape[0]),
            "bold": False,
        },
        {
            "stage": "KLPS-4 (2017-2022)",
            "total": len(klps4),
            "treated": int(klps4[klps4["treated"] == 1].shape[0]),
            "control": int(klps4[klps4["treated"] == 0].shape[0]),
            "bold": True,
        },
        {
            "stage": "  Employment (yes/no)",
            "total": len(emp_n),
            "treated": int(emp_n[emp_n["treated"] == 1].shape[0]),
            "control": int(emp_n[emp_n["treated"] == 0].shape[0]),
            "bold": False,
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

    labels = []
    for row in rows:
        if row["bold"]:
            labels.append(r"$\bf{" + row["stage"] + "}$")
        else:
            labels.append(row["stage"])

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Respondents")
    ax.legend(loc="lower right")
    ax.set_title(TITLE_MAP[1], fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_sample_flow.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 2: Balance table with clustered CIs (split axis)
# ------------------------------------------------------------------
def fig_balance(data, fig_dir):
    klps3 = data["klps3"]

    balance_vars = [
        "age_1998",
        "base_std_ctrl",
        "parent_educ_avg",
        "female",
        "spill_0_3km",
        "spill_3_6km",
    ]
    var_labels = {
        "age_1998": "Age at treatment",
        "base_std_ctrl": "Baseline grade",
        "parent_educ_avg": "Parental education (years)",
        "female": "Female (proportion)",
        "spill_0_3km": "Schools within 3km",
        "spill_3_6km": "Schools 3-6km",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    y = np.arange(len(balance_vars))
    for i, var in enumerate(balance_vars):
        t = klps3[klps3["treated"] == 1][var].dropna()
        c = klps3[klps3["treated"] == 0][var].dropna()
        se_t = cluster_se(klps3[klps3["treated"] == 1], var, "base_schid")
        se_c = cluster_se(klps3[klps3["treated"] == 0], var, "base_schid")
        ax.errorbar(
            t.mean(),
            i + 0.15,
            xerr=1.96 * se_t,
            fmt="o",
            color=PALETTE["treated"],
            capsize=3,
            markersize=5,
            label="Treated" if i == 0 else "",
        )
        ax.errorbar(
            c.mean(),
            i - 0.15,
            xerr=1.96 * se_c,
            fmt="o",
            color=PALETTE["control"],
            capsize=3,
            markersize=5,
            label="Control" if i == 0 else "",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([var_labels[v] for v in balance_vars])
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Mean (95% CI, school-clustered)")
    ax.legend(loc="lower right")
    ax.set_title(TITLE_MAP[2], fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_balance.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 3: BLP coefficients — single panel, all outcomes
# ------------------------------------------------------------------
def fig_blp_coefficients(data, fig_dir):
    blp = data["blp"]
    rw = data["rw"]

    moderators = ["age_1998", "female", "age_x_female"]
    mod_short = {"age_1998": "Age", "female": "Female", "age_x_female": "Age×Female"}
    xlim = (-1.5, 1.0)

    fig, ax = plt.subplots(figsize=(8, 10))

    ytick_pos = []
    ytick_lab = []
    outcome_labels_y = []

    for oi, outcome in enumerate(ALL_OUTCOMES):
        sub_blp = blp[blp["outcome"] == outcome]
        base_y = oi * 7

        outcome_center = base_y + 2.5
        outcome_labels_y.append(outcome_center)

        for j, mod in enumerate(moderators):
            row = sub_blp[sub_blp["coefficient"] == mod].iloc[0]
            rw_row = rw[(rw["outcome"] == outcome) & (rw["coefficient"] == mod)]
            rw_p = rw_row["rw_adjusted_p"].values[0]

            y_unadj = base_y + (2 - j) * 2 + 0.3
            y_rw = base_y + (2 - j) * 2 - 0.3

            ci_lo = max(row["ci_lower"], xlim[0])
            ci_hi = min(row["ci_upper"], xlim[1])
            clipped_left = row["ci_lower"] < xlim[0]
            ax.plot(
                [ci_lo, ci_hi],
                [y_unadj, y_unadj],
                color="#2171b5",
                linewidth=1.5,
                alpha=0.7,
                solid_capstyle="butt",
            )
            if clipped_left:
                ax.plot(ci_lo, y_unadj, "<", color="#2171b5", markersize=4, alpha=0.7)
            ax.plot(row["estimate"], y_unadj, "o", color="#2171b5", markersize=5)

            z_rw = stats.norm.ppf(1 - rw_p / 2) if rw_p < 1 else 0
            if z_rw > 0:
                rw_half = z_rw * row["se"]
                rw_lo_raw = row["estimate"] - rw_half
                rw_hi_raw = row["estimate"] + rw_half
                rw_lo = max(rw_lo_raw, xlim[0])
                rw_hi = min(rw_hi_raw, xlim[1])
                rw_clipped_left = rw_lo_raw < xlim[0]
                ax.plot(
                    [rw_lo, rw_hi],
                    [y_rw, y_rw],
                    color="#cb181d",
                    linewidth=2.5,
                    alpha=0.7,
                    solid_capstyle="butt",
                )
                if rw_clipped_left:
                    ax.plot(rw_lo, y_rw, "<", color="#cb181d", markersize=4, alpha=0.7)
            ax.plot(row["estimate"], y_rw, "s", color="#cb181d", markersize=4)

            ytick_pos.append(y_unadj)
            ytick_lab.append(f"{mod_short[mod]} (unadj)")
            ytick_pos.append(y_rw)
            ytick_lab.append(f"{mod_short[mod]} (RW)")

    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axhline(6.5, color="grey", linewidth=0.3, linestyle=":")
    ax.axhline(13.5, color="grey", linewidth=0.3, linestyle=":")

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=8)
    ax.set_xlim(xlim)

    for oi, outcome in enumerate(ALL_OUTCOMES):
        ax.text(
            xlim[0] + 0.02,
            outcome_labels_y[oi],
            OUTCOME_LABELS[outcome],
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )

    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="#2171b5", label="95% CI (unadjusted)"),
            plt.Line2D([0], [0], color="#cb181d", label="95% CI (Romano-Wolf)"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=10,
    )
    ax.set_xlabel("Coefficient estimate")
    ax.set_title(TITLE_MAP[3], fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0.15, 0.04, 1, 1])
    fig.savefig(fig_dir / "fig3_blp_coefficients.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 4: Outcome distributions by treatment group
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

        if outcome in ("underweight_klps3", "employed_klps4"):
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

        ax.set_xlabel(OUTCOME_LABELS[outcome], fontsize=11)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(TITLE_MAP[4], fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_outcome_distributions.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 5: ATE comparison (DoubleML vs naive OLS)
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
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(TITLE_MAP[5], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_ate_comparison.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 6: MDE visualization (no legend clutter, table notes, correct scale)
# ------------------------------------------------------------------
def fig_mde(data, fig_dir):
    klps3 = data["klps3"]
    bmi = klps3.dropna(subset=["bmi_klps3"])
    treated = bmi[bmi["treated"] == 1]["bmi_klps3"]
    control = bmi[bmi["treated"] == 0]["bmi_klps3"]

    sd_bmi = bmi["bmi_klps3"].std()

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

    icc_scenarios = {
        "Observed (0.023)": 0.023,
        "Conservative (0.05)": 0.05,
        "Very conservative (0.08)": 0.08,
    }
    mde_values = {}
    for label, icc in icc_scenarios.items():
        se = sd_bmi * np.sqrt(
            (1 + (m1 - 1) * icc) / (K1 * m1) + (1 + (m0 - 1) * icc) / (K0 * m0)
        )
        mde = z_sum * se
        mde_values[label] = mde

    ate_bmi = data["ate"].loc[data["ate"]["outcome"] == "bmi_klps3", "ate"].values[0]

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

    mde_conservative = mde_values["Conservative (0.05)"]
    ax.axvspan(
        control_mean - mde_conservative,
        control_mean + mde_conservative,
        alpha=0.15,
        color="#ff7f0e",
        label=f"MDE band (ICC=0.05): +/-{mde_conservative:.2f} BMI",
    )

    ax.annotate(
        f"ATE = {ate_bmi:.2f}",
        xy=(ate_bmi, 0),
        xytext=(ate_bmi + 1.5, ax.get_ylim()[1] * 0.15),
        fontsize=10,
        color="red",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    ax.set_xlabel("BMI (kg/m$^2$)")
    ax.set_ylabel("Density")

    notes_lines = [
        f"Unbalanced CRT: $K_1$={K1} treated schools, $K_0$={K0} control schools; "
        f"average cluster size {m1:.0f} (treated), {m0:.0f} (control). "
        f"Observed ATE = {ate_bmi:.2f} BMI points "
        f"(control mean = {control_mean:.1f}, treated mean = {treated.mean():.1f}). "
        f"MDE at 80% power ($t$-dist, $df$={df_clusters}): "
    ]
    mde_parts = []
    for label, mde_val in mde_values.items():
        mde_parts.append(f"ICC {label}: {mde_val:.2f} BMI ({mde_val / sd_bmi:.2f} SD)")
    notes_lines.append("; ".join(mde_parts) + ".")

    fig.text(
        0.12,
        0.02,
        " ".join(notes_lines),
        fontsize=10,
        ha="left",
        va="bottom",
        style="italic",
    )

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(TITLE_MAP[6], fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(fig_dir / "fig6_mde_visualization.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 7: CATE by moderator tercile (two 2x2 panels)
# ------------------------------------------------------------------
def fig_cate_tercile(data, fig_dir):
    klps3 = data["klps3"]
    klps4 = data["klps4"]

    bmi_uw_outcomes = ["bmi_klps3", "underweight_klps3"]
    educ_emp_outcomes = ["educ_klps3", "employed_klps4"]

    blp_caveat = (
        "Tercile differences are not statistically distinguishable from zero "
        "(BLP p > 0.10 for all moderators, all outcomes)."
    )

    def _draw_panel(outcomes, panel_title, filename):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for oi, outcome in enumerate(outcomes):
            df = klps3 if outcome in OUTCOMES_KLPS3 else klps4
            df = df.dropna(subset=[outcome])
            cate_df = data["cate"][outcome]
            merged = cate_df.merge(df[["pupid", "age_1998", "female"]], on="pupid")

            age_terciles = pd.qcut(
                merged["age_1998"], 3, labels=["Young", "Mid", "Old"]
            )
            merged["age_tercile"] = age_terciles

            ax_hist = axes[0, oi]
            ax_hist.hist(
                merged["cate"],
                bins=40,
                density=True,
                alpha=0.7,
                color="#6baed6",
            )
            ate_val = (
                data["ate"].loc[data["ate"]["outcome"] == outcome, "ate"].values[0]
            )
            ax_hist.axvline(ate_val, color="red", linewidth=1.5, linestyle="--")
            ax_hist.set_xlabel("CATE", fontsize=11)
            ax_hist.set_ylabel("Density", fontsize=11)
            ax_hist.set_title(OUTCOME_LABELS[outcome], fontsize=13, fontweight="bold")
            ax_hist.tick_params(labelsize=10)

            ax_bar = axes[1, oi]
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
            ax_bar.set_xticklabels(labels, fontsize=10)
            ax_bar.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax_bar.set_ylabel("Mean CATE", fontsize=11)
            ax_bar.tick_params(labelsize=10)
            ax_bar.set_xlabel(
                f"{OUTCOME_LABELS[outcome]} — CATE by tercile",
                fontsize=11,
            )

        fig.text(
            0.5,
            0.01,
            blp_caveat,
            ha="center",
            fontsize=11,
            style="italic",
            wrap=True,
        )
        fig.suptitle(panel_title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(fig_dir / filename, dpi=300)
        plt.close(fig)

    _draw_panel(
        bmi_uw_outcomes,
        "Figure 7a: CATE — BMI and underweight",
        "fig7a_cate_tercile_bmi_uw.pdf",
    )
    _draw_panel(
        educ_emp_outcomes,
        "Figure 7b: CATE — Education and employment",
        "fig7b_cate_tercile_educ_emp.pdf",
    )


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

    print("Figure 3: BLP coefficients")
    fig_blp_coefficients(data, fig_dir)

    print("Figure 4: Outcome distributions")
    fig_outcome_distributions(data, fig_dir)

    print("Figure 5: ATE comparison")
    fig_ate_comparison(data, fig_dir)

    print("Figure 6: MDE visualization")
    fig_mde(data, fig_dir)

    print("Figure 7: CATE by tercile")
    fig_cate_tercile(data, fig_dir)

    print(f"All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
