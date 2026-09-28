import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv("statistical_analysis_results.csv")


SHORT_NAMES = {
    "IAM Misconfiguration Score (binned)": "IAM",
    "Encryption Configuration Status": "Encryption",
    "Network Exposure Settings": "Network",
    "Logging Enablement Status": "Logging",
}

df["short_category"] = df["variable_name"].map(SHORT_NAMES)


# ============================================================
# CHART 1
# AVERAGE CRAMER'S V BY SECURITY FACTOR
# ============================================================

avg_v = (
    df.groupby("short_category")["cramers_v"]
    .mean()
    .sort_values(ascending=False)
)


fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    avg_v.index,
    avg_v.values
)


# Add Cramer's V values above bars
for bar, value in zip(bars, avg_v.values):

    ax.annotate(
        f"{value:.3f}",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            value
        ),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10
    )


ax.set_ylabel(
    "Average Cramer's V",
    fontsize=12
)

ax.set_xlabel(
    "Security factor",
    fontsize=11
)

ax.set_title(
    "Association between security factors and "
    "GDPR Article 32 compliance status",
    fontsize=13
)

ax.set_ylim(
    0,
    avg_v.max() * 1.20
)


plt.tight_layout()

plt.savefig(
    "fig_rq1_cramers_v_by_category.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_rq1_cramers_v_by_category.png"
)


# ============================================================
# CHART 2
# CRAMER'S V BY CLOUD PROVIDER
# ============================================================

pivot_v = (
    df.pivot(
        index="short_category",
        columns="provider",
        values="cramers_v"
    )
    .reindex(avg_v.index)
)


providers = [
    "aws",
    "azure",
    "gcp"
]


fig, ax = plt.subplots(
    figsize=(11, 6)
)


x = np.arange(
    len(pivot_v.index)
)

width = 0.25


for i, provider in enumerate(providers):

    positions = x + (i - 1) * width

    values = pivot_v[
        provider
    ].to_numpy()

    bars = ax.bar(
        positions,
        values,
        width,
        label=provider.upper()
    )

    # Display Cramer's V above every bar
    for bar, value in zip(
        bars,
        values
    ):

        ax.annotate(
            f"{value:.3f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                value
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )


ax.set_ylabel(
    "Cramer's V",
    fontsize=12
)

ax.set_xlabel(
    "Security factor",
    fontsize=11
)

ax.set_title(
    "Security-factor associations across "
    "AWS, Azure and GCP",
    fontsize=13
)


ax.set_xticks(x)

ax.set_xticklabels(
    pivot_v.index,
    fontsize=11
)


ax.legend(
    title="Cloud provider"
)


ax.set_ylim(
    0,
    pivot_v.max().max() * 1.20
)


plt.tight_layout()

plt.savefig(
    "fig_rq2_cramers_v_by_provider.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_rq2_cramers_v_by_provider.png"
)


# ============================================================
# CHART 3
# STRONGEST FACTOR BY CLOUD PROVIDER
# ============================================================

strongest_rows = []


for provider in providers:

    provider_df = (
        df[
            df["provider"] == provider
        ]
        .sort_values(
            "cramers_v",
            ascending=False
        )
    )

    strongest = provider_df.iloc[0]

    strongest_rows.append(
        {
            "provider":
                provider.upper(),

            "factor":
                strongest[
                    "short_category"
                ],

            "cramers_v":
                strongest[
                    "cramers_v"
                ],
        }
    )


strongest_df = pd.DataFrame(
    strongest_rows
)


fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


bars = ax.bar(
    strongest_df["provider"],
    strongest_df["cramers_v"]
)


for bar, row in zip(
    bars,
    strongest_df.itertuples()
):

    ax.annotate(
        f"{row.factor}\n"
        f"V = {row.cramers_v:.3f}",
        xy=(
            bar.get_x()
            + bar.get_width() / 2,
            row.cramers_v
        ),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10
    )


ax.set_ylabel(
    "Cramer's V",
    fontsize=12
)

ax.set_xlabel(
    "Cloud provider",
    fontsize=11
)

ax.set_title(
    "Strongest security-factor association "
    "by cloud provider",
    fontsize=13
)


ax.set_ylim(
    0,
    strongest_df["cramers_v"].max() * 1.20
)


plt.tight_layout()

plt.savefig(
    "fig_rq2_strongest_factor_by_provider.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_rq2_strongest_factor_by_provider.png"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n=== Average Cramer's V by security factor ==="
)

for category, value in avg_v.items():

    print(
        f"{category}: {value:.4f}"
    )


print(
    "\n=== Strongest factor by provider ==="
)

for row in strongest_df.itertuples():

    print(
        f"{row.provider}: "
        f"{row.factor} "
        f"(V = {row.cramers_v:.4f})"
    )


print(
    "\nStatistical visualisation completed."
)