import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv("train_set.csv")
test = pd.read_csv("test_set.csv")

feature_cols = [
    col for col in train.columns
    if col != "compliance_status"
]

X_test = test[feature_cols]

# ML target convention:
# 1 = Non-compliant
# 0 = Compliant

y_test = (
    1
    - test["compliance_status"].astype(int)
)


# ============================================================
# LOAD FINAL TRAINED MODELS
# ============================================================

models = {
    "Logistic Regression":
        joblib.load("model_lr.joblib"),

    "Random Forest":
        joblib.load("model_rf.joblib"),

    "XGBoost":
        joblib.load("model_xgb.joblib"),
}


# ============================================================
# RULE-BASED BASELINE
# ============================================================

def rule_based_predict(X):

    iam_fail = (
        X["iam_score"] > 0.5
    )

    enc_fail = (
        X["encryption_status"] == 0
    )

    net_fail = (
        (X["network_exposure_1"] == 1)
        |
        (X["network_exposure_2"] == 1)
    )

    log_fail = (
        X["logging_status"] == 0
    )

    non_compliant = (
        iam_fail
        |
        enc_fail
        |
        net_fail
        |
        log_fail
    )

    # 1 = Non-compliant
    # 0 = Compliant
    return non_compliant.astype(int)


# ============================================================
# 1. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)

for ax, (
    model_name,
    model
) in zip(
    axes[:3],
    models.items()
):

    prediction = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Compliant",
            "Non-compliant"
        ]
    )

    display.plot(
        ax=ax,
        colorbar=False
    )

    ax.set_title(
        model_name
    )


baseline_prediction = (
    rule_based_predict(
        X_test
    )
)

baseline_cm = confusion_matrix(
    y_test,
    baseline_prediction,
    labels=[0, 1]
)

baseline_display = ConfusionMatrixDisplay(
    confusion_matrix=baseline_cm,
    display_labels=[
        "Compliant",
        "Non-compliant"
    ]
)

baseline_display.plot(
    ax=axes[3],
    colorbar=False
)

axes[3].set_title(
    "Rule-Based Baseline"
)

plt.suptitle(
    "Confusion matrices on the held-out test set"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.savefig(
    "fig_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_confusion_matrices.png"
)


# ============================================================
# 2. ROC CURVES
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

for model_name, model in models.items():

    # Probability of class 1 = Non-compliant
    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    roc_auc_value = auc(
        fpr,
        tpr
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC = {roc_auc_value:.3f})"
        )
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC curve comparison for non-compliance detection"
)

ax.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "fig_roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_roc_curves.png"
)


# ============================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_model = (
    models[
        "Random Forest"
    ]
)

rf_importance = (
    pd.Series(
        rf_model.feature_importances_,
        index=feature_cols
    )
    .sort_values(
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

ax.barh(
    rf_importance.index,
    rf_importance.values
)

ax.set_xlabel(
    "Gini importance"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Random Forest feature importance"
)

plt.tight_layout()

plt.savefig(
    "fig_rf_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_rf_feature_importance.png"
)


# ============================================================
# 4. XGBOOST SHAP FEATURE IMPORTANCE
# ============================================================

xgb_model = (
    models[
        "XGBoost"
    ]
)

explainer = (
    shap.TreeExplainer(
        xgb_model
    )
)

shap_values = (
    explainer.shap_values(
        X_test
    )
)

# Compatibility with SHAP versions
# that return a list
if isinstance(
    shap_values,
    list
):
    shap_values = shap_values[-1]


shap_importance = (
    pd.Series(
        np.abs(
            shap_values
        ).mean(
            axis=0
        ),
        index=feature_cols
    )
    .sort_values(
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

ax.barh(
    shap_importance.index,
    shap_importance.values
)

ax.set_xlabel(
    "Mean absolute SHAP value"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "XGBoost SHAP feature importance"
)

plt.tight_layout()

plt.savefig(
    "fig_xgboost_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_xgboost_shap_importance.png"
)


# ============================================================
# 5. MODEL PERFORMANCE COMPARISON
# ============================================================

results = pd.read_csv(
    "model_evaluation_results.csv"
)

# Professor-requested metrics:
# Accuracy
# Precision
# Recall
# F1-score
# ROC-AUC

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

metric_labels = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}


# ------------------------------------------------------------
# Check required columns exist
# ------------------------------------------------------------

required_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

missing_columns = [
    col
    for col in required_columns
    if col not in results.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns in model_evaluation_results.csv: "
        f"{missing_columns}"
    )


# ------------------------------------------------------------
# Prepare chart data
# ------------------------------------------------------------

plot_df = (
    results[
        ["model"] + metric_columns
    ]
    .set_index(
        "model"
    )
    .rename(
        columns=metric_labels
    )
)


# ------------------------------------------------------------
# Plot performance comparison
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 6)
)

plot_df.plot(
    kind="bar",
    ax=ax
)

ax.set_ylabel(
    "Score"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylim(
    0,
    1.05
)

ax.set_title(
    "Model performance comparison"
)

ax.legend(
    title="Metric",
    bbox_to_anchor=(
        1.02,
        1
    ),
    loc="upper left"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "fig_model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved fig_model_performance_comparison.png"
)


# ============================================================
# 6. FINAL MODEL PERFORMANCE TABLE
# ============================================================

print(
    "\n=== Final Model Performance Table ==="
)

display_table = results[
    [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
].copy()


display_table = display_table.rename(
    columns={
        "model":
            "Model",

        "accuracy":
            "Accuracy",

        "precision":
            "Precision",

        "recall":
            "Recall",

        "f1":
            "F1-score",

        "roc_auc":
            "ROC-AUC",
    }
)


# ------------------------------------------------------------
# Print table
# ------------------------------------------------------------

print(
    display_table.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Save table
# ------------------------------------------------------------

display_table.to_csv(
    "final_model_performance_table.csv",
    index=False
)

print(
    "Saved final_model_performance_table.csv"
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print(
    "\n=== Figures generated successfully ==="
)

print(
    "1. fig_confusion_matrices.png"
)

print(
    "2. fig_roc_curves.png"
)

print(
    "3. fig_rf_feature_importance.png"
)

print(
    "4. fig_xgboost_shap_importance.png"
)

print(
    "5. fig_model_performance_comparison.png"
)

print(
    "\nTable generated:"
)

print(
    "final_model_performance_table.csv"
)

print(
    "\nModel visualisation completed."
)