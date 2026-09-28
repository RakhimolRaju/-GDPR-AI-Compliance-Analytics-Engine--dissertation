import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from statsmodels.stats.contingency_tables import mcnemar


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():

    train = pd.read_csv("train_set.csv")
    test = pd.read_csv("test_set.csv")

    feature_cols = [
        col for col in train.columns
        if col != "compliance_status"
    ]

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    # Original dataset:
    # 0 = Non-compliant
    # 1 = Compliant
    #
    # ML modelling:
    # 1 = Non-compliant
    # 0 = Compliant

    y_train = 1 - train["compliance_status"].astype(int)
    y_test = 1 - test["compliance_status"].astype(int)

    print("=== Dataset ===")

    print(
        f"Training set: {len(X_train)} records, "
        f"{X_train.shape[1]} features"
    )

    print(
        f"Training non-compliant rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Test set: {len(X_test)} records, "
        f"{X_test.shape[1]} features"
    )

    print(
        f"Test non-compliant rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
    )


# ============================================================
# 2. CROSS-VALIDATION
# ============================================================

def get_cv():

    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )


# ============================================================
# 3. LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(X_train, y_train):

    print("\n=== Training Logistic Regression ===")

    param_grid = {
        "C": [
            0.001,
            0.01,
            0.1,
            1,
            10,
        ],

        "class_weight": [
            None,
            "balanced",
        ],
    }

    model = LogisticRegression(
        penalty="l2",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=get_cv(),
        n_jobs=-1,
        refit=True
    )

    grid.fit(
        X_train,
        y_train
    )

    print(
        f"Best parameters: "
        f"{grid.best_params_}"
    )

    print(
        f"Best cross-validation F1: "
        f"{grid.best_score_:.4f}"
    )

    return (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_,
    )


# ============================================================
# 4. RANDOM FOREST
# ============================================================

def train_random_forest(X_train, y_train):

    print("\n=== Training Random Forest ===")

    param_grid = {
        "n_estimators": [
            100,
            200,
            500,
        ],

        "max_depth": [
            None,
            10,
            20,
            30,
        ],

        "min_samples_split": [
            2,
            5,
            10,
        ],

        "class_weight": [
            None,
            "balanced",
        ],
    }

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=get_cv(),
        n_jobs=-1,
        refit=True
    )

    grid.fit(
        X_train,
        y_train
    )

    print(
        f"Best parameters: "
        f"{grid.best_params_}"
    )

    print(
        f"Best cross-validation F1: "
        f"{grid.best_score_:.4f}"
    )

    return (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_,
    )


# ============================================================
# 5. XGBOOST
# ============================================================

def train_xgboost(X_train, y_train):

    print("\n=== Training XGBoost ===")

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    class_ratio = (
        negative_count / positive_count
    )

    print(
        f"Compliant/non-compliant class ratio: "
        f"{class_ratio:.4f}"
    )

    param_grid = {
        "learning_rate": [
            0.01,
            0.05,
            0.1,
            0.2,
        ],

        "max_depth": [
            3,
            5,
            7,
            10,
        ],

        "n_estimators": [
            100,
            300,
            500,
        ],

        "subsample": [
            0.8,
            1.0,
        ],

        "scale_pos_weight": [
            1.0,
            class_ratio,
        ],
    }

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=get_cv(),
        n_jobs=-1,
        refit=True
    )

    grid.fit(
        X_train,
        y_train
    )

    print(
        f"Best parameters: "
        f"{grid.best_params_}"
    )

    print(
        f"Best cross-validation F1: "
        f"{grid.best_score_:.4f}"
    )

    return (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_,
    )


# ============================================================
# 6. RULE-BASED BASELINE
# ============================================================

def rule_based_baseline(X):

    iam_fail = (
        X["iam_score"] > 0.5
    )

    encryption_fail = (
        X["encryption_status"] == 0
    )

    network_fail = (
        (X["network_exposure_1"] == 1)
        |
        (X["network_exposure_2"] == 1)
    )

    logging_fail = (
        X["logging_status"] == 0
    )

    non_compliant = (
        iam_fail
        |
        encryption_fail
        |
        network_fail
        |
        logging_fail
    )

    # 1 = Non-compliant
    # 0 = Compliant

    return non_compliant.astype(int)


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    y_probability=None,
    model_name=""
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if y_probability is not None:

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

    else:

        roc_auc = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print(
        f"\n--- {model_name} ---"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    if not np.isnan(roc_auc):

        print(
            f"ROC-AUC:   {roc_auc:.4f}"
        )

    else:

        print(
            "ROC-AUC:   n/a"
        )

    print(
        "Confusion matrix:"
    )

    print(cm)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# 8. MCNEMAR TEST
# ============================================================

def mcnemar_test(
    y_true,
    pred_a,
    pred_b,
    name_a,
    name_b
):

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = (
        pred_a == y_true
    )

    correct_b = (
        pred_b == y_true
    )

    both_correct = (
        correct_a
        & correct_b
    ).sum()

    a_only = (
        correct_a
        & ~correct_b
    ).sum()

    b_only = (
        ~correct_a
        & correct_b
    ).sum()

    both_wrong = (
        ~correct_a
        & ~correct_b
    ).sum()

    table = [
        [
            both_correct,
            a_only,
        ],
        [
            b_only,
            both_wrong,
        ],
    ]

    exact_test = (
        a_only + b_only
        < 25
    )

    result = mcnemar(
        table,
        exact=exact_test,
        correction=not exact_test
    )

    print(
        f"\nMcNemar's test: "
        f"{name_a} vs {name_b}"
    )

    print(
        f"{name_a} only correct: "
        f"{a_only}"
    )

    print(
        f"{name_b} only correct: "
        f"{b_only}"
    )

    print(
        f"Statistic: "
        f"{result.statistic:.4f}"
    )

    print(
        f"P-value: "
        f"{result.pvalue:.4f}"
    )

    if result.pvalue < 0.05:

        print(
            "Significant difference"
        )

    else:

        print(
            "No significant difference"
        )

    return {
        "comparison":
            f"{name_a} vs {name_b}",

        "a_only":
            int(a_only),

        "b_only":
            int(b_only),

        "statistic":
            float(result.statistic),

        "p_value":
            float(result.pvalue),

        "significant":
            result.pvalue < 0.05,
    }


# ============================================================
# 9. MAIN
# ============================================================

def main():

    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
    ) = load_data()

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    (
        lr_model,
        lr_params,
        lr_cv_f1,
    ) = train_logistic_regression(
        X_train,
        y_train
    )

    (
        rf_model,
        rf_params,
        rf_cv_f1,
    ) = train_random_forest(
        X_train,
        y_train
    )

    (
        xgb_model,
        xgb_params,
        xgb_cv_f1,
    ) = train_xgboost(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    joblib.dump(
        lr_model,
        "model_lr.joblib"
    )

    joblib.dump(
        rf_model,
        "model_rf.joblib"
    )

    joblib.dump(
        xgb_model,
        "model_xgb.joblib"
    )

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    tuning_results = pd.DataFrame(
        [
            {
                "model":
                    "Logistic Regression",

                "best_cv_f1":
                    lr_cv_f1,

                "best_parameters":
                    str(lr_params),
            },

            {
                "model":
                    "Random Forest",

                "best_cv_f1":
                    rf_cv_f1,

                "best_parameters":
                    str(rf_params),
            },

            {
                "model":
                    "XGBoost",

                "best_cv_f1":
                    xgb_cv_f1,

                "best_parameters":
                    str(xgb_params),
            },
        ]
    )

    tuning_results.to_csv(
        "hyperparameter_tuning_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Evaluate final models
    # --------------------------------------------------------

    results = []
    predictions = {}

    models = [
        (
            "Logistic Regression",
            lr_model
        ),

        (
            "Random Forest",
            rf_model
        ),

        (
            "XGBoost",
            xgb_model
        ),
    ]

    for model_name, model in models:

        prediction = model.predict(
            X_test
        )

        probability = model.predict_proba(
            X_test
        )[:, 1]

        predictions[
            model_name
        ] = prediction

        results.append(
            evaluate_model(
                y_test,
                prediction,
                probability,
                model_name
            )
        )

    # --------------------------------------------------------
    # Rule-based baseline
    # --------------------------------------------------------

    baseline_prediction = (
        rule_based_baseline(
            X_test
        )
    )

    predictions[
        "Rule-Based Baseline"
    ] = baseline_prediction

    results.append(
        evaluate_model(
            y_test,
            baseline_prediction,
            None,
            "Rule-Based Baseline"
        )
    )

    # --------------------------------------------------------
    # Save final evaluation
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "model_evaluation_results.csv",
        index=False
    )

    print(
        "\n=== Final Model Comparison ==="
    )

    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    print(
        "\n=== Random Forest Feature Importance ==="
    )

    rf_importance = pd.Series(
        rf_model.feature_importances_,
        index=feature_cols,
        name="importance"
    ).sort_values(
        ascending=False
    )

    print(
        rf_importance.to_string()
    )

    rf_importance.to_csv(
        "rf_feature_importance.csv"
    )

    # ========================================================
    # XGBOOST SHAP IMPORTANCE
    # ========================================================

    print(
        "\n=== XGBoost SHAP Feature Importance ==="
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer.shap_values(
        X_test
    )

    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[-1]
        )

    mean_abs_shap = pd.Series(
        np.abs(
            shap_values
        ).mean(
            axis=0
        ),

        index=feature_cols,

        name="mean_abs_shap"
    ).sort_values(
        ascending=False
    )

    print(
        mean_abs_shap.to_string()
    )

    mean_abs_shap.to_csv(
        "xgboost_shap_importance.csv"
    )

    # ========================================================
    # MCNEMAR TESTS
    # ========================================================

    print(
        "\n=== McNemar's Test Comparisons ==="
    )

    comparisons = [
        (
            "Logistic Regression",
            "Rule-Based Baseline"
        ),

        (
            "Random Forest",
            "Rule-Based Baseline"
        ),

        (
            "XGBoost",
            "Rule-Based Baseline"
        ),

        (
            "Logistic Regression",
            "Random Forest"
        ),

        (
            "Logistic Regression",
            "XGBoost"
        ),

        (
            "Random Forest",
            "XGBoost"
        ),
    ]

    mcnemar_results = []

    for model_a, model_b in comparisons:

        result = mcnemar_test(
            y_test.to_numpy(),

            predictions[
                model_a
            ],

            predictions[
                model_b
            ],

            model_a,
            model_b
        )

        mcnemar_results.append(
            result
        )

    pd.DataFrame(
        mcnemar_results
    ).to_csv(
        "mcnemar_results.csv",
        index=False
    )

    print(
        "\n=== Model training completed ==="
    )


if __name__ == "__main__":
    main()