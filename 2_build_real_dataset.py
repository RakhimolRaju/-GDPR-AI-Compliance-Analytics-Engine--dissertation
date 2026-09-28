

import pandas as pd

# ---- Category mapping rules ---------------------------------------------
# Each check is assigned to exactly one of the four IVs based on
# SERVICE_NAME first, falling back to keyword matches in CHECK_ID for
# services that span more than one category (e.g. 'ec2' covers both
# encryption (EBS) and network exposure (security groups)).

IAM_SERVICES = {"iam", "accessanalyzer", "organizations", "account"}
LOGGING_SERVICES = {"cloudtrail", "config", "cloudwatch", "eventbridge", "logging", "monitor"}
ENCRYPTION_SERVICES = {"kms", "s3", "backup", "storage", "cloudstorage"}
NETWORK_SERVICES = {"vpc", "networkfirewall", "network"}

# ec2/gcp-gemini etc. need keyword-level disambiguation
ENCRYPTION_KEYWORDS = ["encrypt", "kms", "ebs_default", "ebs_volume_encryption"]
NETWORK_KEYWORDS = ["securitygroup", "networkacl", "exposed_to_internet", "public_ip",
                     "internet_facing", "wide_open", "firewall", "internet_allowed",
                     "internet_access"]
IAM_KEYWORDS = ["iam_", "password_policy", "mfa", "root_", "access_key", "accesskey", "os_login"]
LOGGING_KEYWORDS = ["logging", "log_metric", "audit_log", "config_recorder", "flow_logs", "flow_log"]


def categorize(row):
    service = str(row["SERVICE_NAME"]).lower()
    check_id = str(row["CHECK_ID"]).lower()

    if service in ENCRYPTION_SERVICES:
        return "encryption"
    if service in NETWORK_SERVICES:
        return "network_exposure"
    if service in IAM_SERVICES:
        return "iam"
    if service in LOGGING_SERVICES:
        return "logging"

    # ec2 and other cross-cutting services: disambiguate by check_id keywords
    if any(k in check_id for k in ENCRYPTION_KEYWORDS):
        return "encryption"
    if any(k in check_id for k in NETWORK_KEYWORDS):
        return "network_exposure"
    if any(k in check_id for k in IAM_KEYWORDS):
        return "iam"
    if any(k in check_id for k in LOGGING_KEYWORDS):
        return "logging"

    return "uncategorized"


def load_and_tag(path, provider):
    df = pd.read_csv(path, sep=";")
    df["provider"] = provider
    df["category"] = df.apply(categorize, axis=1)
    return df


def build_resource_records(df):
    """
    Pivot from long (one row per check) to wide (one row per resource),
    with one status column per category. Where a resource has multiple
    checks in the same category, take the WORST result (FAIL beats PASS)
    
    """
    df = df[df["category"] != "uncategorized"].copy()
    df["fail_flag"] = (df["STATUS"] == "FAIL").astype(int)

    # one row per resource per category = worst-case (max) fail flag
    pivoted = (
        df.groupby(["provider", "RESOURCE_UID", "RESOURCE_TYPE", "REGION", "category"])["fail_flag"]
        .max()
        .unstack("category")
        .reset_index()
    )

    for col in ["iam", "encryption", "network_exposure", "logging"]:
        if col not in pivoted.columns:
            pivoted[col] = pd.NA  # category wasn't present for any resource

    # Rename the raw category fail-flag columns up front to avoid a name
    # collision with the final output column of the same name below.
    pivoted = pivoted.rename(columns={
        "RESOURCE_UID": "record_id",
        "RESOURCE_TYPE": "resource_type",
        "REGION": "region",
        "network_exposure": "network_exposure_raw",
    })

    # IMPORTANT: do NOT fillna(0) here. A resource that was never assessed
    # for a given category (e.g. an Access Analyzer resource has no
    # encryption checks at all) must stay NaN/missing -- treating "not
    # measured" as "compliant" would silently fabricate data. This is a
    # genuine structural property of raw CSPM output: Prowler only runs
    # checks relevant to each resource's own service, so most individual
    # resources are naturally assessed in ONE category, not all four.
    pivoted["iam_score"] = pivoted["iam"]  # NaN = not assessed, 0 = pass, 1 = fail
    pivoted["encryption_status"] = pivoted["encryption"].map({0: 1, 1: 0})  # 1=encrypted/compliant, 0=fail, NaN=not assessed
    pivoted["network_exposure"] = pivoted["network_exposure_raw"]  # 0=compliant, 1=exposed, NaN=not assessed
    pivoted["logging_status"] = pivoted["logging"].map({0: 1, 1: 0})  # 1=logging enabled/compliant, 0=fail, NaN=not assessed

    # Count how many of the 4 categories actually have real data for this resource
    pivoted["categories_measured"] = pivoted[["iam", "encryption", "network_exposure_raw", "logging"]].notna().sum(axis=1)

    # Overall compliance: only computed where at least one category was
    # measured, and only among the categories that WERE measured (an
    # unmeasured category cannot count as a violation).
    fail_any = (
        (pivoted["iam"] == 1)
        | (pivoted["encryption"] == 1)
        | (pivoted["network_exposure_raw"] == 1)
        | (pivoted["logging"] == 1)
    )
    pivoted["compliance_status"] = [
        pd.NA if cm == 0 else int(not fa)
        for cm, fa in zip(pivoted["categories_measured"], fail_any)
    ]

    pivoted["data_source"] = "real"

    keep = ["record_id", "provider", "resource_type", "region", "iam_score",
            "encryption_status", "network_exposure", "logging_status",
            "categories_measured", "compliance_status", "data_source"]
    return pivoted[keep]


def main():
    aws = load_and_tag("Prowler\\prowler-output-914820856546-20260806124711.csv", "aws")
    azure = load_and_tag("Prowler\\prowler-output-f03d8550-6d93-42c7-8699-c0c73c7da435-20260806181536.csv", "azure")
    gcp = load_and_tag("Prowler\\prowler-gcp-output-default-20260806180129.csv", "gcp")

    combined_raw = pd.concat([aws, azure, gcp], ignore_index=True)

    print("=== Category coverage by provider (raw check counts) ===")
    print(combined_raw.groupby(["provider", "category"]).size().unstack(fill_value=0))
    print()

    records = build_resource_records(combined_raw)

    print("=== Final resource-level records ===")
    print("Total records:", len(records))
    print(records.groupby("provider").size())
    print()
    print("=== Non-null coverage per IV (out of total records) ===")
    for col in ["iam_score", "encryption_status", "network_exposure", "logging_status"]:
        print(f"{col}: {records[col].notna().sum()} / {len(records)}")

    records.to_csv(records.to_csv("real_evidence_dataset.csv", index=False))
    print("\nSaved to real_evidence_dataset.csv")

    # Also print real fail rates per provider per category -- these feed
    # directly into build_final_dataset.py's calibration step.
    print("\n=== Real fail rates per provider per category (for calibration) ===")
    combined_raw["fail"] = (combined_raw["STATUS"] == "FAIL").astype(int)
    sub = combined_raw[combined_raw["category"] != "uncategorized"]
    print(sub.groupby(["provider", "category"])["fail"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
