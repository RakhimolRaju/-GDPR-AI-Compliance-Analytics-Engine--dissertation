

import argparse
import gzip
import json
import os
from collections import defaultdict

import pandas as pd

# Events considered IAM-privilege-relevant for the misconfiguration score
IAM_SENSITIVE_EVENTS = {
    "CreateUser", "CreateAccessKey", "AttachUserPolicy", "AttachRolePolicy",
    "PutUserPolicy", "PutRolePolicy", "CreatePolicyVersion", "AssumeRole",
    "CreateRole", "UpdateAssumeRolePolicy", "AddUserToGroup",
    "CreateLoginProfile", "UpdateLoginProfile", "DeactivateMFADevice",
    "ConsoleLogin", "GetSessionToken",
}

MFA_RELATED_EVENTS = {"DeactivateMFADevice", "EnableMFADevice", "ConsoleLogin"}


def iter_events(filepath):
    """Yield individual CloudTrail event dicts from a .json or .json.gz file."""
    opener = gzip.open if filepath.endswith(".gz") else open
    try:
        with opener(filepath, "rt", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    # CloudTrail files are usually {"Records": [ ... ]}
    if isinstance(data, dict) and "Records" in data:
        for event in data["Records"]:
            yield event
    elif isinstance(data, list):
        for event in data:
            yield event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="cloudtrail_iam_summary.csv")
    args = parser.parse_args()

    per_user_stats = defaultdict(lambda: {
        "total_events": 0,
        "iam_sensitive_events": 0,
        "denied_events": 0,
        "mfa_related_events": 0,
        "console_logins_no_mfa": 0,
        "distinct_source_ips": set(),
        "access_key_used": 0,
        "assumed_role_used": 0,
    })

    files_processed = 0
    total_events_seen = 0

    for fname in os.listdir(args.input_dir):
        if not (fname.endswith(".json") or fname.endswith(".json.gz")):
            continue
        filepath = os.path.join(args.input_dir, fname)
        files_processed += 1

        for event in iter_events(filepath):
            total_events_seen += 1
            identity = event.get("userIdentity", {}) or {}
            user_key = identity.get("arn") or identity.get("accountId") or identity.get("principalId") or "unknown"

            stats = per_user_stats[user_key]
            stats["total_events"] += 1

            event_name = event.get("eventName", "")
            if event_name in IAM_SENSITIVE_EVENTS:
                stats["iam_sensitive_events"] += 1
            if event_name in MFA_RELATED_EVENTS:
                stats["mfa_related_events"] += 1
            if event_name == "ConsoleLogin":
                additional = event.get("additionalEventData", {}) or {}
                if additional.get("MFAUsed") == "No":
                    stats["console_logins_no_mfa"] += 1

            if event.get("errorCode"):
                stats["denied_events"] += 1

            src_ip = event.get("sourceIPAddress")
            if src_ip:
                stats["distinct_source_ips"].add(src_ip)

            identity_type = identity.get("type", "")
            if identity_type == "IAMUser":
                stats["access_key_used"] += 1
            elif identity_type == "AssumedRole":
                stats["assumed_role_used"] += 1

        if files_processed % 50 == 0:
            print(f"Processed {files_processed} files, {total_events_seen} events so far...")

    print(f"\nDone. Processed {files_processed} files, {total_events_seen} total events, {len(per_user_stats)} distinct identities.")

    rows = []
    for user_key, s in per_user_stats.items():
        total = max(s["total_events"], 1)
        rows.append({
            "identity": user_key,
            "total_events": s["total_events"],
            "iam_sensitive_event_rate": round(s["iam_sensitive_events"] / total, 4),
            "denied_event_rate": round(s["denied_events"] / total, 4),
            "console_logins_no_mfa": s["console_logins_no_mfa"],
            "distinct_source_ip_count": len(s["distinct_source_ips"]),
            "used_iam_user_credentials": int(s["access_key_used"] > 0),
            "used_assumed_role": int(s["assumed_role_used"] > 0),
        })

    summary_df = pd.DataFrame(rows)

    # Simple composite IAM Misconfiguration Score (0-1), combining the
    # signals above -- weighted toward MFA absence and privilege-sensitive
    
    summary_df["iam_misconfiguration_score"] = (
        0.4 * summary_df["iam_sensitive_event_rate"].clip(0, 1)
        + 0.3 * (summary_df["console_logins_no_mfa"] > 0).astype(int)
        + 0.2 * summary_df["denied_event_rate"].clip(0, 1)
        + 0.1 * (summary_df["distinct_source_ip_count"] > 3).astype(int)
    ).round(4)

    summary_df.to_csv(args.output, index=False)
    print(f"Saved summary for {len(summary_df)} identities to {args.output}")
    #print("\nThis file is small enough to upload directly -- the raw logs do not need to leave your machine.")


if __name__ == "__main__":
    main()
