
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    print(
        f"Loaded {len(df)} records from {path}"
    )

    return df


# ============================================================
# 2. REMOVE ROWS WITH MISSING TARGET
# ============================================================

def remove_missing_target(df):
    """
    Remove records where compliance_status is missing.
    """

    before = len(df)

    df = df.dropna(
        subset=["compliance_status"]
    ).copy()

    dropped = before - len(df)

    print(
        "\n=== Step 1: Missing target handling ==="
    )

    print(
        f"Records dropped because compliance_status was missing: "
        f"{dropped}"
    )

    return df


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """
    Remove exact duplicate observations.

    record_id is excluded because it is unique by definition.
    """

    check_cols = [
        col
        for col in df.columns
        if col != "record_id"
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=check_cols
    ).copy()

    removed = before - len(df)

    print(
        "\n=== Step 2: Duplicate removal ==="
    )

    print(
        f"Exact duplicate records removed: {removed}"
    )

    return df


# ============================================================
# 4. LABEL CONSISTENCY VALIDATION
# ============================================================

def step_label_validation(
    real_evidence_path,
    n_sample=100
):
    """
    Prepare a real-evidence sample for manual validation.

    If label_validation_sample.csv already contains a complete
    manual_label column, Cohen's Kappa is calculated automatically.

    manual_label:
        0 = Non-compliant
        1 = Compliant
    """

    print(
        "\n=== Step 3: Label consistency validation ==="
    )

    validation_file = (
        "label_validation_sample.csv"
    )

    # --------------------------------------------------------
    # Check whether a previous validation sample already exists
    # --------------------------------------------------------

    if os.path.exists(validation_file):

        reviewed_df = pd.read_csv(
            validation_file
        )

        if "manual_label" in reviewed_df.columns:

            valid_rows = reviewed_df[
                reviewed_df[
                    "manual_label"
                ].notna()
            ].copy()

            total_sample = len(
                reviewed_df
            )

            reviewed_count = len(
                valid_rows
            )

           
            if (
                reviewed_count > 0
                and reviewed_count < total_sample
            ):

                print(
                    f"Manual review incomplete: "
                    f"{reviewed_count}/{total_sample} "
                    f"records labelled."
                )

                print(
                    "Cohen's Kappa will be calculated "
                    "after all sampled records have been reviewed."
                )

                return None

            # ------------------------------------------------
            # Full manual review completed
            # ------------------------------------------------

            if (
                reviewed_count == total_sample
                and total_sample > 0
            ):

                # Validate manual labels
                unique_labels = set(
                    valid_rows[
                        "manual_label"
                    ].astype(int).unique()
                )

                if not unique_labels.issubset(
                    {0, 1}
                ):

                    raise ValueError(
                        "manual_label must contain only "
                        "0 or 1 values."
                    )

                valid_rows[
                    "manual_label"
                ] = (
                    valid_rows[
                        "manual_label"
                    ]
                    .astype(int)
                )

                valid_rows[
                    "compliance_status"
                ] = (
                    valid_rows[
                        "compliance_status"
                    ]
                    .astype(int)
                )

                kappa = (
                    cohen_kappa_score(
                        valid_rows[
                            "compliance_status"
                        ],
                        valid_rows[
                            "manual_label"
                        ]
                    )
                )

                print(
                    f"Reviewed records: "
                    f"{reviewed_count}"
                )

                print(
                    f"Cohen's Kappa: "
                    f"{kappa:.4f}"
                )

                if kappa >= 0.80:

                    print(
                        "Label agreement: strong"
                    )

                elif kappa >= 0.60:

                    print(
                        "Label agreement: substantial"
                    )

                elif kappa >= 0.40:

                    print(
                        "Label agreement: moderate"
                    )

                else:

                    print(
                        "Label agreement: low"
                    )

                return kappa

    # --------------------------------------------------------
    # Create validation sample if a reviewed sample does not exist
    # --------------------------------------------------------

    real_df = pd.read_csv(
        real_evidence_path
    )

    reviewable = real_df[
        real_df[
            "compliance_status"
        ].notna()
    ].copy()

    if len(reviewable) == 0:

        print(
            "No labelled real records are available "
            "for validation."
        )

        return None

    sample_n = min(
        n_sample,
        len(reviewable)
    )

    sample = reviewable.sample(
        n=sample_n,
        random_state=RANDOM_STATE
    )

    # Add an empty manual review column
    sample[
        "manual_label"
    ] = np.nan

    sample.to_csv(
        validation_file,
        index=False
    )

    print(
        f"Validation sample created: "
        f"{validation_file}"
    )

    print(
        f"Sample size: {sample_n} records"
    )

    return None


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_raw_data(df):
    """
    Perform a stratified 80/20 train-test split.
    """

    train_df, test_df = (
        train_test_split(
            df,
            test_size=0.20,
            stratify=df[
                "compliance_status"
            ],
            random_state=RANDOM_STATE
        )
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    print(
        "\n=== Step 4: Stratified 80/20 train-test split ==="
    )

    print(
        f"Raw train set: "
        f"{len(train_df)} records, "
        f"{train_df['compliance_status'].mean() * 100:.2f}% compliant"
    )

    print(
        f"Raw test set:  "
        f"{len(test_df)} records, "
        f"{test_df['compliance_status'].mean() * 100:.2f}% compliant"
    )

    return train_df, test_df


# ============================================================
# 6. MISSING VALUE IMPUTATION
# ============================================================

def impute_using_training_data(
    train_df,
    test_df
):
    """
    Learn imputation values from the training set and apply
    the same values to the test set.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    numeric_cols = [
        "iam_score"
    ]

    categorical_cols = [
        "encryption_status",
        "network_exposure",
        "logging_status",
        "provider"
    ]

    print(
        "\n=== Step 5: Missing-value imputation ==="
    )

    # --------------------------------------------------------
    # Numeric variables
    # --------------------------------------------------------

    for col in numeric_cols:

        train_missing = (
            train_df[col]
            .isna()
            .sum()
        )

        test_missing = (
            test_df[col]
            .isna()
            .sum()
        )

        median_value = (
            train_df[col]
            .median()
        )

        train_df[col] = (
            train_df[col]
            .fillna(
                median_value
            )
        )

        test_df[col] = (
            test_df[col]
            .fillna(
                median_value
            )
        )

        print(
            f"{col}: "
            f"median = {median_value:.4f}, "
            f"train imputed = {train_missing}, "
            f"test imputed = {test_missing}"
        )

    # --------------------------------------------------------
    # Categorical variables
    # --------------------------------------------------------

    for col in categorical_cols:

        train_missing = (
            train_df[col]
            .isna()
            .sum()
        )

        test_missing = (
            test_df[col]
            .isna()
            .sum()
        )

        mode_value = (
            train_df[col]
            .mode()
            .iloc[0]
        )

        train_df[col] = (
            train_df[col]
            .fillna(
                mode_value
            )
        )

        test_df[col] = (
            test_df[col]
            .fillna(
                mode_value
            )
        )

        print(
            f"{col}: "
            f"mode = {mode_value}, "
            f"train imputed = {train_missing}, "
            f"test imputed = {test_missing}"
        )

    return train_df, test_df


# ============================================================
# 7. IAM MIN-MAX NORMALISATION
# ============================================================

def normalise_iam_using_training_data(
    train_df,
    test_df
):
    """
    Min-max normalise iam_score using the training-set range.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_min = (
        train_df[
            "iam_score"
        ].min()
    )

    train_max = (
        train_df[
            "iam_score"
        ].max()
    )

    print(
        "\n=== Step 6: IAM score normalisation ==="
    )

    print(
        f"Training IAM range before scaling: "
        f"[{train_min:.4f}, {train_max:.4f}]"
    )

    if train_max > train_min:

        train_df[
            "iam_score"
        ] = (
            train_df[
                "iam_score"
            ]
            - train_min
        ) / (
            train_max
            - train_min
        )

        test_df[
            "iam_score"
        ] = (
            test_df[
                "iam_score"
            ]
            - train_min
        ) / (
            train_max
            - train_min
        )

    else:

        train_df[
            "iam_score"
        ] = 0.0

        test_df[
            "iam_score"
        ] = 0.0

    print(
        "Training IAM range after scaling: "
        f"[{train_df['iam_score'].min():.4f}, "
        f"{train_df['iam_score'].max():.4f}]"
    )

    print(
        "Test IAM range after scaling: "
        f"[{test_df['iam_score'].min():.4f}, "
        f"{test_df['iam_score'].max():.4f}]"
    )

    return train_df, test_df


# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

def encode_using_training_categories(
    train_df,
    test_df
):
    """
    One-hot encode network exposure and provider.

    The categories present in the training set define the
    encoded feature structure.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    print(
        "\n=== Step 7: One-hot encoding ==="
    )

    categorical_cols = [
        "network_exposure",
        "provider"
    ]

    training_categories = {}

    for col in categorical_cols:

        training_categories[col] = (
            sorted(
                train_df[col]
                .dropna()
                .unique()
                .tolist()
            )
        )

        print(
            f"{col} categories: "
            f"{training_categories[col]}"
        )

    # --------------------------------------------------------
    # Apply training-set categories to both datasets
    # --------------------------------------------------------

    for col in categorical_cols:

        categories = (
            training_categories[col]
        )

        train_df[col] = (
            pd.Categorical(
                train_df[col],
                categories=categories
            )
        )

        test_df[col] = (
            pd.Categorical(
                test_df[col],
                categories=categories
            )
        )

    # --------------------------------------------------------
    # One-hot encode
    # --------------------------------------------------------

    train_df = pd.get_dummies(
        train_df,
        columns=categorical_cols,
        prefix=categorical_cols,
        dtype=int
    )

    test_df = pd.get_dummies(
        test_df,
        columns=categorical_cols,
        prefix=categorical_cols,
        dtype=int
    )

    # --------------------------------------------------------
    # Align test columns with training columns
    # --------------------------------------------------------

    for col in train_df.columns:

        if col not in test_df.columns:

            test_df[col] = 0

    test_df = test_df.reindex(
        columns=train_df.columns,
        fill_value=0
    )

    encoded_cols = [
        col
        for col in train_df.columns
        if (
            col.startswith(
                "network_exposure_"
            )
            or col.startswith(
                "provider_"
            )
        )
    ]

    print(
        f"Encoded columns: "
        f"{encoded_cols}"
    )

    return train_df, test_df


# ============================================================
# 9. SELECT FINAL ML FEATURES
# ============================================================

def select_features(
    train_df,
    test_df
):
    """
    Remove metadata fields that are not used as ML predictors.
    """

    exclude_cols = [
        "record_id",
        "resource_type",
        "region",
        "data_source"
    ]

    train_df = train_df.drop(
        columns=[
            col
            for col in exclude_cols
            if col in train_df.columns
        ]
    )

    test_df = test_df.drop(
        columns=[
            col
            for col in exclude_cols
            if col in test_df.columns
        ]
    )

    feature_cols = [
        col
        for col in train_df.columns
        if col != "compliance_status"
    ]

    train_df = train_df[
        feature_cols
        + ["compliance_status"]
    ]

    test_df = test_df[
        feature_cols
        + ["compliance_status"]
    ]

    return train_df, test_df


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

def final_checks(
    train_df,
    test_df
):
    """
    Perform final preprocessing checks before saving.
    """

    print(
        "\n=== Step 8: Final validation ==="
    )

    if (
        list(train_df.columns)
        != list(test_df.columns)
    ):

        raise ValueError(
            "Train and test column structures do not match."
        )

    train_missing = (
        train_df
        .isna()
        .sum()
        .sum()
    )

    test_missing = (
        test_df
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values in train set: "
        f"{train_missing}"
    )

    print(
        f"Missing values in test set: "
        f"{test_missing}"
    )

    if (
        train_missing > 0
        or test_missing > 0
    ):

        raise ValueError(
            "Missing values remain after preprocessing."
        )

    print(
        f"Train set: "
        f"{len(train_df)} records, "
        f"{train_df['compliance_status'].mean() * 100:.2f}% compliant"
    )

    print(
        f"Test set:  "
        f"{len(test_df)} records, "
        f"{test_df['compliance_status'].mean() * 100:.2f}% compliant"
    )

    print(
        f"Final columns: "
        f"{list(train_df.columns)}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    df = load_data(
        "gdpr_compliance_training_dataset.csv"
    )

    # --------------------------------------------------------
    # Initial cleaning
    # --------------------------------------------------------

    df = remove_missing_target(
        df
    )

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # Real-evidence label validation
    # --------------------------------------------------------

    step_label_validation(
        "real_evidence_dataset.csv",
        n_sample=100
    )

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    train_df, test_df = (
        split_raw_data(
            df
        )
    )

    # --------------------------------------------------------
    # Missing-value imputation
    # --------------------------------------------------------

    train_df, test_df = (
        impute_using_training_data(
            train_df,
            test_df
        )
    )

    # --------------------------------------------------------
    # IAM normalisation
    # --------------------------------------------------------

    train_df, test_df = (
        normalise_iam_using_training_data(
            train_df,
            test_df
        )
    )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    train_df, test_df = (
        encode_using_training_categories(
            train_df,
            test_df
        )
    )

    # --------------------------------------------------------
    # Final feature selection
    # --------------------------------------------------------

    train_df, test_df = (
        select_features(
            train_df,
            test_df
        )
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    final_checks(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # Save processed datasets
    # --------------------------------------------------------

    train_df.to_csv(
        "train_set.csv",
        index=False
    )

    test_df.to_csv(
        "test_set.csv",
        index=False
    )

    print(
        "\n=== Final output ==="
    )

    print(
        f"train_set.csv: "
        f"{len(train_df)} records, "
        f"{len(train_df.columns)} columns"
    )

    print(
        f"test_set.csv: "
        f"{len(test_df)} records, "
        f"{len(test_df.columns)} columns"
    )

    print(
        "\nPreprocessing completed successfully."
    )


if __name__ == "__main__":
    main()

