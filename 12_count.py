
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train_set.csv")

print(
    train[
        ["provider_aws", "provider_azure", "provider_gcp"]
    ].sum()
)


# ============================================================
# LOAD FINAL TRAIN AND TEST DATASETS
# ============================================================

train_df = pd.read_csv("train_set.csv")
test_df = pd.read_csv("test_set.csv")

# Combine them to reconstruct the final 2,797-record dataset
df = pd.concat([train_df, test_df], ignore_index=True)

# ============================================================
# CLASS DISTRIBUTION
# compliance_status:
# 0 = Non-compliant
# 1 = Compliant
# ============================================================

non_compliant = (df["compliance_status"] == 0).sum()
compliant = (df["compliance_status"] == 1).sum()

counts = [non_compliant, compliant]

labels = [
    f"Non-compliant\n({non_compliant} records)",
    f"Compliant\n({compliant} records)"
]

# ============================================================
# CREATE PIE CHART
# ============================================================

plt.figure(figsize=(10, 8))

plt.pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 16}
)

plt.title(
    "Article 32 compliance-status distribution",
    fontsize=22
)

plt.tight_layout()

# ============================================================
# SAVE OUTPUT
# ============================================================

plt.savefig(
    "fig_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n=== Article 32 Compliance-Status Distribution ===")
print(f"Non-compliant: {non_compliant}")
print(f"Compliant:     {compliant}")
print(f"Total:         {len(df)}")
print("\nSaved: fig_class_distribution.png")