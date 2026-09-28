

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.05

REAL_FILE = "real_evidence_dataset.csv"
SYNTHETIC_FILE = "gdpr_compliance_training_dataset.csv"

OUTPUT_CSV = "real_vs_synthetic_validation.csv"
OUTPUT_FIGURE = "real_vs_synthetic_validation_chart.png"


# ============================================================
# LOAD DATA
# ============================================================

real_df = pd.read_csv(REAL_FILE)
synthetic_df = pd.read_csv(SYNTHETIC_FILE)

print("=== Dataset Loading ===")

print(
    f"Real evidence records loaded: "
    f"{len(real_df)}"
)

print(
    f"Synthetic dataset records loaded: "
    f"{len(synthetic_df)}"
)


# ============================================================
# KEEP SYNTHETIC CALIBRATED RECORDS
# ============================================================

if "data_source" in synthetic_df.columns:

    synthetic_df = synthetic_df[
        synthetic_df["data_source"]
        == "synthetic_calibrated"
    ].copy()

    print(
        "Synthetic calibrated records retained: "
        f"{len(synthetic_df)}"
    )

else:

    print(
        "Column 'data_source' not found. "
        "Using all records from synthetic dataset."
    )


# ============================================================
# NORMALISE PROVIDER VALUES
# ============================================================

real_df["provider"] = (
    real_df["provider"]
    .astype(str)
    .str.strip()
    .str.lower()
)

synthetic_df["provider"] = (
    synthetic_df["provider"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

real_before = len(real_df)

real_df = (
    real_df
    .drop_duplicates()
    .reset_index(drop=True)
)

real_removed = (
    real_before
    - len(real_df)
)


synthetic_before = len(synthetic_df)

synthetic_df = (
    synthetic_df
    .drop_duplicates()
    .reset_index(drop=True)
)

synthetic_removed = (
    synthetic_before
    - len(synthetic_df)
)


print("\n=== Duplicate Removal ===")

print(
    f"Real exact duplicates removed: "
    f"{real_removed}"
)

print(
    f"Synthetic exact duplicates removed: "
    f"{synthetic_removed}"
)

print(
    f"Real records after duplicate removal: "
    f"{len(real_df)}"
)

print(
    f"Synthetic records after duplicate removal: "
    f"{len(synthetic_df)}"
)


# ============================================================
# SECURITY FAILURE DEFINITIONS
# ============================================================

# IAM:
#     score > 0.50 = failure
#
# Encryption:
#     0 = failure
#
# Network:
#     0 = no exposure
#     1 = moderate/restricted exposure
#     2 = high/unrestricted exposure
#
#     Both 1 and 2 are treated as exposure failures.
#
# Logging:
#     0 = failure


FAIL_RULES = {

    "iam_score":
        lambda series:
            pd.to_numeric(
                series,
                errors="coerce"
            ) > 0.5,

    "encryption_status":
        lambda series:
            pd.to_numeric(
                series,
                errors="coerce"
            ) == 0,

    "network_exposure":
        lambda series:
            pd.to_numeric(
                series,
                errors="coerce"
            ) > 0,

    "logging_status":
        lambda series:
            pd.to_numeric(
                series,
                errors="coerce"
            ) == 0,
}


LABELS = {

    "iam_score":
        "IAM Misconfiguration",

    "encryption_status":
        "Encryption Failure",

    "network_exposure":
        "Network Exposure",

    "logging_status":
        "Logging Failure",
}


SHORT_NAMES = {

    "IAM Misconfiguration":
        "IAM",

    "Encryption Failure":
        "Encryption",

    "Network Exposure":
        "Network",

    "Logging Failure":
        "Logging",
}


PROVIDERS = [
    "aws",
    "azure",
    "gcp"
]


# ============================================================
# HELPER:
# DECIDE WHICH STATISTICAL TEST TO USE
# ============================================================

def compare_proportions(
    real_fails,
    real_n,
    synthetic_fails,
    synthetic_n
):

    """
    Compare two failure proportions.

    Uses the two-proportion z-test when expected
    cell counts are sufficiently large.

    Falls back to Fisher's Exact Test when
    any 2x2 cell count is below 5.
    """

    real_passes = (
        real_n
        - real_fails
    )

    synthetic_passes = (
        synthetic_n
        - synthetic_fails
    )

    contingency = np.array(
        [
            [
                real_fails,
                real_passes
            ],
            [
                synthetic_fails,
                synthetic_passes
            ]
        ]
    )


    # --------------------------------------------
    # SMALL-CELL CHECK
    # --------------------------------------------

    if (
        contingency < 5
    ).any():

        _, p_value = (
            fisher_exact(
                contingency
            )
        )

        return (
            np.nan,
            p_value,
            "Fisher's Exact"
        )


    # --------------------------------------------
    # TWO-PROPORTION Z-TEST
    # --------------------------------------------

    count = np.array(
        [
            real_fails,
            synthetic_fails
        ]
    )

    nobs = np.array(
        [
            real_n,
            synthetic_n
        ]
    )

    statistic, p_value = (
        proportions_ztest(
            count,
            nobs
        )
    )

    return (
        statistic,
        p_value,
        "Two-proportion z-test"
    )


# ============================================================
# RUN VALIDATION
# ============================================================

results = []


print(
    "\n=== Real vs Synthetic Validation ===\n"
)


for variable, fail_function in FAIL_RULES.items():

    # --------------------------------------------
    # CHECK REQUIRED COLUMN EXISTS
    # --------------------------------------------

    if variable not in real_df.columns:

        print(
            f"Skipping {variable}: "
            "column missing from real dataset."
        )

        continue


    if variable not in synthetic_df.columns:

        print(
            f"Skipping {variable}: "
            "column missing from synthetic dataset."
        )

        continue


    for provider in PROVIDERS:

        # ----------------------------------------
        # REAL SUBSET
        # ----------------------------------------

        real_subset = real_df[
            (
                real_df["provider"]
                == provider
            )
            &
            (
                real_df[variable]
                .notna()
            )
        ].copy()


        # ----------------------------------------
        # SYNTHETIC SUBSET
        # ----------------------------------------

        synthetic_subset = synthetic_df[
            (
                synthetic_df["provider"]
                == provider
            )
            &
            (
                synthetic_df[variable]
                .notna()
            )
        ].copy()


        real_n = (
            len(real_subset)
        )

        synthetic_n = (
            len(synthetic_subset)
        )


        # ----------------------------------------
        # NO REAL EVIDENCE
        # ----------------------------------------

        if real_n == 0:

            print(
                f"Skipping "
                f"{LABELS[variable]} / "
                f"{provider.upper()}: "
                "no measured observations."
            )

            continue


        if synthetic_n == 0:

            print(
                f"Skipping "
                f"{LABELS[variable]} / "
                f"{provider.upper()}: "
                "no synthetic observations."
            )

            continue


        # ----------------------------------------
        # FAILURE COUNTS
        # ----------------------------------------

        real_fail_mask = (
            fail_function(
                real_subset[
                    variable
                ]
            )
        )

        synthetic_fail_mask = (
            fail_function(
                synthetic_subset[
                    variable
                ]
            )
        )


        real_fails = int(
            real_fail_mask.sum()
        )

        synthetic_fails = int(
            synthetic_fail_mask.sum()
        )


        # ----------------------------------------
        # FAILURE RATES
        # ----------------------------------------

        real_rate = (
            real_fails
            / real_n
        )

        synthetic_rate = (
            synthetic_fails
            / synthetic_n
        )

        absolute_difference = abs(
            real_rate
            - synthetic_rate
        )


        # ----------------------------------------
        # STATISTICAL TEST
        # ----------------------------------------

        try:

            statistic, p_value, test_used = (
                compare_proportions(
                    real_fails,
                    real_n,
                    synthetic_fails,
                    synthetic_n
                )
            )

            note = ""

        except Exception as error:

            statistic = np.nan
            p_value = np.nan
            test_used = "Not computed"

            note = (
                f"Statistical test failed: "
                f"{error}"
            )


        results.append(
            {
                "category":
                    LABELS[
                        variable
                    ],

                "provider":
                    provider.upper(),

                "real_n":
                    real_n,

                "real_failures":
                    real_fails,

                "real_fail_rate":
                    real_rate,

                "synthetic_n":
                    synthetic_n,

                "synthetic_failures":
                    synthetic_fails,

                "synthetic_fail_rate":
                    synthetic_rate,

                "absolute_difference":
                    absolute_difference,

                "test_used":
                    test_used,

                "test_statistic":
                    statistic,

                "raw_p_value":
                    p_value,

                "note":
                    note,
            }
        )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise ValueError(
        "No valid real-versus-synthetic "
        "comparisons could be performed."
    )


# ============================================================
# HOLM MULTIPLE-COMPARISON ADJUSTMENT
# ============================================================

results_df[
    "holm_adjusted_p"
] = np.nan


valid_mask = (
    results_df[
        "raw_p_value"
    ].notna()
)


valid_p_values = (
    results_df.loc[
        valid_mask,
        "raw_p_value"
    ]
    .values
)


if len(valid_p_values) > 0:

    reject, adjusted_p, _, _ = (
        multipletests(
            valid_p_values,
            alpha=ALPHA,
            method="holm"
        )
    )


    results_df.loc[
        valid_mask,
        "holm_adjusted_p"
    ] = adjusted_p


    results_df.loc[
        valid_mask,
        "significant_difference"
    ] = reject


else:

    results_df[
        "significant_difference"
    ] = np.nan


# ============================================================
# INTERPRETATION COLUMN
# ============================================================

def interpret_result(row):

    if pd.isna(
        row["holm_adjusted_p"]
    ):

        return (
            "Test not available"
        )


    if (
        row[
            "holm_adjusted_p"
        ]
        < ALPHA
    ):

        return (
            "Statistically significant "
            "difference detected"
        )


    return (
        "No statistically significant "
        "difference detected"
    )


results_df[
    "interpretation"
] = results_df.apply(
    interpret_result,
    axis=1
)


# ============================================================
# ROUND DISPLAY VALUES
# ============================================================

for column in [
    "real_fail_rate",
    "synthetic_fail_rate",
    "absolute_difference",
    "test_statistic",
    "raw_p_value",
    "holm_adjusted_p",
]:

    results_df[
        column
    ] = (
        results_df[
            column
        ]
        .round(4)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "=== Real vs Synthetic Security-Control "
    "Failure Rates ===\n"
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

valid_results = results_df[
    results_df[
        "holm_adjusted_p"
    ].notna()
]


no_difference_count = (
    (
        valid_results[
            "holm_adjusted_p"
        ]
        >= ALPHA
    )
    .sum()
)


significant_difference_count = (
    (
        valid_results[
            "holm_adjusted_p"
        ]
        < ALPHA
    )
    .sum()
)


print(
    "\n=== Validation Summary ==="
)

print(
    f"Valid statistical comparisons: "
    f"{len(valid_results)}"
)

print(
    "No statistically significant difference "
    f"after Holm correction: "
    f"{no_difference_count}"
)

print(
    "Statistically significant difference "
    f"after Holm correction: "
    f"{significant_difference_count}"
)


print(
    "\nInterpretation:"
)

print(
    "A non-significant result indicates that "
    "this analysis did not detect statistical "
    "evidence of a difference between the "
    "measured and synthetic failure rates."
)

print(
    "It should not be interpreted as proof "
    "that the two distributions are identical."
)


# ============================================================
# CREATE VISUALISATION DATA
# ============================================================

plot_df = (
    results_df.copy()
)

plot_df[
    "short_category"
] = (
    plot_df[
        "category"
    ]
    .map(
        SHORT_NAMES
    )
)

plot_df[
    "label"
] = (
    plot_df[
        "short_category"
    ]
    + " ("
    + plot_df[
        "provider"
    ]
    + ")"
)


# ============================================================
# GROUPED BAR CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        14,
        7
    )
)


x = np.arange(
    len(
        plot_df
    )
)

width = 0.36


bars_real = ax.bar(
    x - width / 2,

    plot_df[
        "real_fail_rate"
    ],

    width,

    label=(
        "Measured evidence"
    )
)


bars_synthetic = ax.bar(
    x + width / 2,

    plot_df[
        "synthetic_fail_rate"
    ],

    width,

    label=(
        "Synthetic calibrated data"
    )
)


# ============================================================
# ANNOTATE ADJUSTED P-VALUES
# ============================================================

for index, row in plot_df.iterrows():

    adjusted_p = (
        row[
            "holm_adjusted_p"
        ]
    )


    if pd.isna(
        adjusted_p
    ):

        annotation = (
            "N/A"
        )

    elif adjusted_p < 0.001:

        annotation = (
            "p(adj)<.001"
        )

    else:

        annotation = (
            f"p(adj)="
            f"{adjusted_p:.3f}"
        )


    y_position = (
        max(
            row[
                "real_fail_rate"
            ],

            row[
                "synthetic_fail_rate"
            ]
        )
        + 0.035
    )


    ax.annotate(
        annotation,

        xy=(
            index,
            y_position
        ),

        ha="center",

        va="bottom",

        fontsize=8
    )


# ============================================================
# FORMAT FIGURE
# ============================================================

ax.set_ylabel(
    "Security-control failure rate"
)

ax.set_xlabel(
    "Security factor and cloud provider"
)

ax.set_title(
    "Measured and Synthetic Security-Control Failure Rates"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    plot_df[
        "label"
    ],

    rotation=35,

    ha="right"
)


ax.set_ylim(
    0,
    1.15
)


ax.legend()


plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    f"\nSaved {OUTPUT_CSV}"
)

print(
    f"Saved {OUTPUT_FIGURE}"
)

print(
    "\nReal-versus-synthetic validation completed."
)