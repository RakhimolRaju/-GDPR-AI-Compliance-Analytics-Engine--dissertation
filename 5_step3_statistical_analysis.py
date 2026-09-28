
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


RANDOM_STATE = 42


# ============================================================
# 1. CRAMER'S V
# ============================================================

def cramers_v(chi2, n, r, c):
    

    if n == 0:
        return np.nan

    denominator = min(r - 1, c - 1)

    if denominator <= 0:
        return np.nan

    phi2 = chi2 / n

    return np.sqrt(
        phi2 / denominator
    )


# ============================================================
# 2. INTERPRET CRAMER'S V
# ============================================================

def interpret_v(v):
    """
    Interpret Cramer's V using conventional effect-size bands.
    """

    if pd.isna(v):
        return "n/a"

    if v < 0.10:
        return "negligible"

    elif v < 0.30:
        return "small"

    elif v < 0.50:
        return "medium"

    else:
        return "large"


# ============================================================
# 3. RECONSTRUCT ORIGINAL CATEGORICAL VARIABLES
# ============================================================

def reconstruct_columns(df):
    """
    Reconstruct provider and network_exposure categories
    from the one-hot encoded columns.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    provider_cols = [
        "provider_aws",
        "provider_azure",
        "provider_gcp",
    ]

    network_cols = [
        "network_exposure_0",
        "network_exposure_1",
        "network_exposure_2",
    ]

    required_cols = (
        provider_cols
        + network_cols
    )

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # --------------------------------------------------------
    # Reconstruct provider
    # --------------------------------------------------------
    #
    # Example:
    #
    # provider_aws   provider_azure   provider_gcp
    #      1               0               0
    #
    # becomes:
    #
    # provider = aws
    #
    # idxmax() returns the column containing the highest value.
    # --------------------------------------------------------

    df["provider"] = (
        df[provider_cols]
        .astype(float)
        .idxmax(axis=1)
        .str.replace(
            "provider_",
            "",
            regex=False
        )
    )

    # --------------------------------------------------------
    # Reconstruct network exposure
    # --------------------------------------------------------

    df["network_exposure_cat"] = (
        df[network_cols]
        .astype(float)
        .idxmax(axis=1)
        .str.replace(
            "network_exposure_",
            "",
            regex=False
        )
        .astype(int)
    )

    return df


# ============================================================
# 4. IAM SCORE BANDING
# ============================================================

def bin_iam_score(df):
    """
    Discretise IAM score into low, medium and high bands.
    """

    df = df.copy()

    bins = [
        -0.001,
        0.33,
        0.66,
        1.0
    ]

    labels = [
        "low",
        "medium",
        "high"
    ]

    df["iam_band"] = pd.cut(
        df["iam_score"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


# ============================================================
# 5. ASSOCIATION TEST
# ============================================================

def run_association_test(
    df,
    variable_col,
    provider
):
    """
    Test association between one security-control variable
    and compliance_status for one cloud provider.

    Chi-square is used by default.

    If a 2x2 contingency table has an expected count below 5,
    Fisher's Exact Test is used for the p-value.

    Cramer's V is reported as the effect-size measure.
    """

    # --------------------------------------------------------
    # Select provider
    # --------------------------------------------------------

    sub = df[
        df["provider"] == provider
    ].copy()

    # --------------------------------------------------------
    # Build contingency table
    # --------------------------------------------------------

    contingency = pd.crosstab(
        sub[variable_col],
        sub["compliance_status"]
    )

    # --------------------------------------------------------
    # Check sufficient variation
    # --------------------------------------------------------

    if (
        contingency.shape[0] < 2
        or contingency.shape[1] < 2
    ):

        return {
            "provider": provider,
            "variable": variable_col,
            "n": len(sub),
            "chi2": np.nan,
            "p_value": np.nan,
            "significant_p<0.05": False,
            "cramers_v": np.nan,
            "test_used": "n/a",
            "min_expected_count": np.nan,
            "interpretation": "insufficient variation",
        }

    # --------------------------------------------------------
    # Chi-square test
    # --------------------------------------------------------

    chi2, p_value, dof, expected = (
        chi2_contingency(
            contingency
        )
    )

    min_expected = (
        expected.min()
    )

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    v = cramers_v(
        chi2,
        len(sub),
        contingency.shape[0],
        contingency.shape[1]
    )

    test_used = "Chi-square"

    reliability_note = ""

    # --------------------------------------------------------
    # Fisher's Exact fallback
    # --------------------------------------------------------

    if min_expected < 5:

        if contingency.shape == (2, 2):

            _, p_fisher = (
                fisher_exact(
                    contingency.to_numpy()
                )
            )

            p_value = p_fisher

            test_used = (
                "Fisher's Exact"
            )

        else:

            reliability_note = (
                f"CAUTION: minimum expected count "
                f"{min_expected:.2f} < 5"
            )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    interpretation = (
        interpret_v(v)
    )

    if reliability_note:

        interpretation = (
            interpretation
            + " -- "
            + reliability_note
        )

    return {
        "provider": provider,
        "variable": variable_col,
        "n": len(sub),

        "chi2": round(
            chi2,
            3
        ),

        "p_value": p_value,

        "significant_p<0.05": (
            p_value < 0.05
        ),

        "cramers_v": (
            round(v, 4)
            if not pd.isna(v)
            else np.nan
        ),

        "test_used": test_used,

        "min_expected_count": round(
            min_expected,
            2
        ),

        "interpretation": interpretation,
    }


# ============================================================
# 6. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load TRAINING DATA ONLY
    # --------------------------------------------------------

    df = pd.read_csv(
        "train_set.csv"
    )

    print(
        f"Training dataset for statistical analysis: "
        f"{len(df)} records"
    )

    print(
        f"Compliance rate: "
        f"{df['compliance_status'].mean() * 100:.2f}%\n"
    )

    # --------------------------------------------------------
    # Reconstruct original categories
    # --------------------------------------------------------

    df = reconstruct_columns(
        df
    )

    # --------------------------------------------------------
    # IAM score bands
    # --------------------------------------------------------

    df = bin_iam_score(
        df
    )

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    variables = {

        "iam_band":
            "IAM Misconfiguration Score (binned)",

        "encryption_status":
            "Encryption Configuration Status",

        "network_exposure_cat":
            "Network Exposure Settings",

        "logging_status":
            "Logging Enablement Status",
    }

    providers = [
        "aws",
        "azure",
        "gcp"
    ]

    # --------------------------------------------------------
    # Run 12 association tests
    # --------------------------------------------------------

    results = []

    for variable_col, variable_name in variables.items():

        for provider in providers:

            result = (
                run_association_test(
                    df,
                    variable_col,
                    provider
                )
            )

            result[
                "variable_name"
            ] = variable_name

            results.append(
                result
            )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # 7. HOLM MULTIPLE-TESTING CORRECTION
    # ========================================================

    results_df[
        "p_value_holm"
    ] = np.nan

    valid_mask = (
        results_df[
            "p_value"
        ].notna()
    )

    if valid_mask.sum() > 0:

        corrected_p = multipletests(
            results_df.loc[
                valid_mask,
                "p_value"
            ],
            alpha=0.05,
            method="holm"
        )[1]

        results_df.loc[
            valid_mask,
            "p_value_holm"
        ] = corrected_p

    results_df[
        "significant_holm_p<0.05"
    ] = (
        results_df[
            "p_value_holm"
        ] < 0.05
    )

    # --------------------------------------------------------
    # Final output columns
    # --------------------------------------------------------

    results_df = results_df[
        [
            "variable_name",
            "provider",
            "n",
            "chi2",
            "p_value",
            "significant_p<0.05",
            "p_value_holm",
            "significant_holm_p<0.05",
            "cramers_v",
            "test_used",
            "min_expected_count",
            "interpretation",
        ]
    ]

    # --------------------------------------------------------
    # Display settings
    # --------------------------------------------------------

    pd.set_option(
        "display.width",
        200
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.max_rows",
        None
    )

    # --------------------------------------------------------
    # Print complete statistical table
    # --------------------------------------------------------

    print(
        "=== Statistical association results ==="
    )

    print(
        "(12 tests: 4 security factors x "
        "3 cloud providers)\n"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    results_df.to_csv(
        "statistical_analysis_results.csv",
        index=False
    )

    print(
        "\nSaved to "
        "statistical_analysis_results.csv"
    )

    # ========================================================
    # 8. RQ1
    # ========================================================

    print(
        "\n=== RQ1: Average Cramer's V "
        "per security factor ==="
    )

    average_v = (
        results_df
        .groupby(
            "variable_name"
        )[
            "cramers_v"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        average_v.to_string()
    )

    if len(average_v) > 0:

        print(
            f"\nStrongest association overall: "
            f"{average_v.index[0]}"
        )

    # ========================================================
    # 9. RQ2
    # ========================================================

    print(
        "\n=== RQ2: Cramer's V "
        "by cloud provider ==="
    )

    provider_pivot = (
        results_df
        .pivot(
            index="variable_name",
            columns="provider",
            values="cramers_v"
        )
    )

    print(
        provider_pivot.to_string()
    )

    # --------------------------------------------------------
    # Strongest factor for each cloud provider
    # --------------------------------------------------------

    print(
        "\n=== Strongest security factor "
        "for each cloud provider ==="
    )

    for provider in providers:

        provider_results = (
            results_df[
                results_df[
                    "provider"
                ] == provider
            ]
            .dropna(
                subset=[
                    "cramers_v"
                ]
            )
            .sort_values(
                "cramers_v",
                ascending=False
            )
        )

        if len(provider_results) == 0:

            print(
                f"{provider.upper()}: "
                f"No valid result"
            )

            continue

        strongest = (
            provider_results.iloc[0]
        )

        print(
            f"{provider.upper()}: "
            f"{strongest['variable_name']} "
            f"(V = {strongest['cramers_v']:.4f})"
        )


if __name__ == "__main__":
    main()
