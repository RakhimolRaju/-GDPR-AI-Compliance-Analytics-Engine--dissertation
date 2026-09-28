import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


RANDOM_STATE = 42

N_BOOTSTRAPS = 1000

CONFIDENCE_LEVEL = 0.95


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    test = pd.read_csv(
        "test_set.csv"
    )

    feature_cols = [
        col for col in test.columns
        if col != "compliance_status"
    ]

    X_test = test[
        feature_cols
    ]

    # ML convention:
    # 1 = Non-compliant
    # 0 = Compliant

    y_test = (
        1
        - test[
            "compliance_status"
        ].astype(int)
    )

    print(
        "=== Test Data ==="
    )

    print(
        f"Test records: "
        f"{len(test)}"
    )

    print(
        f"Compliant: "
        f"{(y_test == 0).sum()}"
    )

    print(
        f"Non-compliant: "
        f"{(y_test == 1).sum()}"
    )

    return (
        X_test,
        y_test
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    return {
        "Logistic Regression":
            joblib.load(
                "model_lr.joblib"
            ),

        "Random Forest":
            joblib.load(
                "model_rf.joblib"
            ),

        "XGBoost":
            joblib.load(
                "model_xgb.joblib"
            ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1_score":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_probability
            ),
    }


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_model(
    model_name,
    model,
    X_test,
    y_test
):

    print(
        f"\n=== Bootstrapping "
        f"{model_name} ==="
    )

    y_pred = (
        model.predict(
            X_test
        )
    )

    y_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    original_metrics = (
        calculate_metrics(
            y_test,
            y_pred,
            y_probability
        )
    )


    rng = (
        np.random.default_rng(
            RANDOM_STATE
        )
    )


    y_array = (
        y_test.to_numpy()
    )


    compliant_indices = (
        np.where(
            y_array == 0
        )[0]
    )


    noncompliant_indices = (
        np.where(
            y_array == 1
        )[0]
    )


    bootstrap_results = []


    for iteration in range(
        N_BOOTSTRAPS
    ):

        sample_compliant = (
            rng.choice(
                compliant_indices,
                size=len(
                    compliant_indices
                ),
                replace=True
            )
        )


        sample_noncompliant = (
            rng.choice(
                noncompliant_indices,
                size=len(
                    noncompliant_indices
                ),
                replace=True
            )
        )


        bootstrap_indices = (
            np.concatenate(
                [
                    sample_compliant,
                    sample_noncompliant
                ]
            )
        )


        rng.shuffle(
            bootstrap_indices
        )


        metrics = calculate_metrics(

            y_array[
                bootstrap_indices
            ],

            y_pred[
                bootstrap_indices
            ],

            y_probability[
                bootstrap_indices
            ]
        )


        metrics[
            "iteration"
        ] = (
            iteration + 1
        )


        bootstrap_results.append(
            metrics
        )


    return (
        original_metrics,
        pd.DataFrame(
            bootstrap_results
        )
    )


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def confidence_interval(
    values
):

    alpha = (
        1
        - CONFIDENCE_LEVEL
    )


    lower = np.percentile(
        values,
        100 * alpha / 2
    )


    upper = np.percentile(
        values,
        100 * (
            1 - alpha / 2
        )
    )


    return (
        lower,
        upper
    )


# ============================================================
# MAIN
# ============================================================

def main():

    X_test, y_test = (
        load_test_data()
    )


    models = (
        load_models()
    )


    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]


    summary_results = []

    all_iterations = []


    for (
        model_name,
        model
    ) in models.items():


        (
            original_metrics,
            bootstrap_df
        ) = bootstrap_model(
            model_name,
            model,
            X_test,
            y_test
        )


        bootstrap_df.insert(
            0,
            "model",
            model_name
        )


        all_iterations.append(
            bootstrap_df
        )


        print(
            f"\n=== {model_name}: "
            f"95% Confidence Intervals ==="
        )


        for metric in metric_names:

            lower, upper = (
                confidence_interval(
                    bootstrap_df[
                        metric
                    ]
                )
            )


            point_estimate = (
                original_metrics[
                    metric
                ]
            )


            print(
                f"{metric:<10} "
                f"{point_estimate:.4f} "
                f"(95% CI: "
                f"{lower:.4f} - "
                f"{upper:.4f})"
            )


            summary_results.append(
                {
                    "model":
                        model_name,

                    "metric":
                        metric,

                    "point_estimate":
                        point_estimate,

                    "ci_lower":
                        lower,

                    "ci_upper":
                        upper,
                }
            )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary_df = (
        pd.DataFrame(
            summary_results
        )
    )


    all_iterations_df = (
        pd.concat(
            all_iterations,
            ignore_index=True
        )
    )


    summary_df.to_csv(
        "bootstrap_confidence_intervals.csv",
        index=False
    )


    all_iterations_df.to_csv(
        "bootstrap_all_iterations.csv",
        index=False
    )


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    display_df = (
        summary_df.copy()
    )


    display_df[
        "result"
    ] = display_df.apply(

        lambda row:

        f"{row['point_estimate']:.4f} "
        f"({row['ci_lower']:.4f} - "
        f"{row['ci_upper']:.4f})",

        axis=1
    )


    final_table = (
        display_df.pivot(
            index="model",
            columns="metric",
            values="result"
        )
    )


    final_table = (
        final_table.reindex(
            columns=[
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        )
    )


    print(
        "\n=== Final Bootstrap Evaluation ==="
    )


    print(
        final_table.to_string()
    )


    final_table.to_csv(
        "bootstrap_summary_table.csv"
    )


    print(
        "\nBootstrap evaluation completed."
    )


    print(
        "\nFiles saved:"
    )

    print(
        "bootstrap_confidence_intervals.csv"
    )

    print(
        "bootstrap_summary_table.csv"
    )

    print(
        "bootstrap_all_iterations.csv"
    )


if __name__ == "__main__":
    main()