import pandas as pd

df = pd.read_csv("real_evidence_dataset.csv")

# column, whether 1 already means "fail" (True) or "compliant" (False)
CATEGORY_COLUMNS = {
    "iam": ("iam_score", True),
    "encryption": ("encryption_status", False),
    "network_exposure": ("network_exposure", True),
    "logging": ("logging_status", False),
}

for category, (col, one_means_fail) in CATEGORY_COLUMNS.items():
    measured = df[df[col].notna()]
    pass_rate = measured.groupby("provider")[col].mean()
    fail_rate = pass_rate if one_means_fail else (1 - pass_rate)
    print(f"\n{category} (n measured per provider):")
    print(fail_rate.round(4))
    print(measured.groupby("provider")[col].count())