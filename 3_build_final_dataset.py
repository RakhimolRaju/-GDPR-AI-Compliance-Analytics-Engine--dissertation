
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# Real, measured fail rates extracted from actual Prowler scans
# ---------------------------------------------------------------------
# These values are preserved from the original evidence-calibrated
# dataset generation approach.
REAL_RATES = {
    "aws": {
        "iam": 0.818,
        "encryption": 0.833,
        "network_exposure": 0.320,
        "logging": 0.746,
    },
    "azure": {
        "iam": 0.409,
        "encryption": 0.750,
        "network_exposure": 0.730,
        "logging": 0.937,
    },
    "gcp": {
        "iam": 0.909,
        "encryption": 0.917,
        "network_exposure": 0.600,
        "logging": 1.000,
    },
}

# ---------------------------------------------------------------------
# Real CloudTrail behavioural IAM data
# ---------------------------------------------------------------------
CLOUDTRAIL_SCORES = pd.read_csv(
    "cloudtrail_iam_summary.csv"
)["iam_misconfiguration_score"].values


RESOURCE_TYPES = {
    "aws": [
        "AwsS3Bucket",
        "AwsIamUser",
        "AwsIamRole",
        "AwsEc2SecurityGroup",
        "AwsEc2Instance",
        "AwsCloudTrailTrail",
        "AwsKmsKey",
        "AwsEc2Vpc",
    ],
    "azure": [
        "StorageAccount",
        "VirtualMachine",
        "NetworkSecurityGroup",
        "KeyVault",
        "SQLDatabase",
        "ActivityLogAlert",
        "RoleAssignment",
    ],
    "gcp": [
        "ComputeInstance",
        "CloudStorageBucket",
        "IAMServiceAccount",
        "VPCFirewallRule",
        "CloudSQLInstance",
        "LogSink",
    ],
}


REGIONS = {
    "aws": [
        "eu-central-1",
        "eu-west-1",
        "us-east-1",
        "us-west-2",
        "ap-southeast-1",
        "ap-northeast-1",
        "sa-east-1",
        "ca-central-1",
        "eu-north-1",
    ],
    "azure": [
        "westeurope",
        "northeurope",
        "eastus",
        "westus2",
        "southeastasia",
    ],
    "gcp": [
        "europe-west1",
        "us-central1",
        "asia-east1",
        "us-east1",
        "europe-north1",
    ],
}


# ---------------------------------------------------------------------
# Sigmoid function
# ---------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------
# Generate evidence-calibrated synthetic records
# ---------------------------------------------------------------------
def generate_provider_records(provider, n_records):

    rates = REAL_RATES[provider]
    records = []

    for i in range(n_records):

        # -------------------------------------------------------------
        # Environmental risk factor
        # -------------------------------------------------------------
        # Introduces moderate correlation between different cloud
        # security weaknesses.
        env_risk = np.random.beta(2, 3)

        def sample_failure(base_rate):
            probability = (
                0.70 * base_rate
                + 0.30 * env_risk
            )

            return int(
                np.random.random() < probability
            )

        # -------------------------------------------------------------
        # Latent control failures
        # -------------------------------------------------------------
        iam_fail = sample_failure(rates["iam"])
        enc_fail = sample_failure(rates["encryption"])
        net_fail = sample_failure(rates["network_exposure"])
        log_fail = sample_failure(rates["logging"])

        # -------------------------------------------------------------
        # IAM score
        # -------------------------------------------------------------
        if provider == "aws":

            raw_val = float(
                np.random.choice(CLOUDTRAIL_SCORES)
            )

            percentile = float(
                (CLOUDTRAIL_SCORES <= raw_val).mean()
            )

            if iam_fail:

                iam_score = round(
                    0.51
                    + 0.48 * percentile,
                    3,
                )

            else:

                iam_score = round(
                    0.01
                    + 0.48 * (1 - percentile),
                    3,
                )

        else:

            iam_score = round(
                np.clip(
                    np.random.beta(
                        2 + 3 * iam_fail,
                        2 + 3 * (1 - iam_fail),
                    ),
                    0,
                    1,
                ),
                3,
            )

        # -------------------------------------------------------------
        # Observable security-control variables
        # -------------------------------------------------------------
        encryption_status = (
            0 if enc_fail else 1
        )

        network_exposure = (
            2
            if (
                net_fail
                and np.random.random() < 0.4
            )
            else (
                1 if net_fail else 0
            )
        )

        logging_status = (
            0 if log_fail else 1
        )

        

        risk_score = (
            0.30 * iam_fail
            + 0.25 * enc_fail
            + 0.20 * net_fail
            + 0.25 * log_fail
        )

        # -------------------------------------------------------------
        # Non-linear interaction effects
        # -------------------------------------------------------------
        # Multiple simultaneous control weaknesses may increase
        # compliance risk more than isolated weaknesses.
        risk_score += (
            0.12
            * iam_fail
            * log_fail
        )

        risk_score += (
            0.10
            * enc_fail
            * net_fail
        )

        # -------------------------------------------------------------
        # Provider-related variation
        # -------------------------------------------------------------
        # Small differences represent provider-specific operational
        # complexity without making provider deterministically define
        # compliance.
        provider_adjustment = {
            "aws": 0.00,
            "azure": 0.03,
            "gcp": 0.05,
        }

        risk_score += (
            provider_adjustment[provider]
        )

        # -------------------------------------------------------------
        # Unobserved / measurement uncertainty
        # -------------------------------------------------------------
        risk_score += np.random.normal(
            loc=0.0,
            scale=0.08,
        )

        # -------------------------------------------------------------
        # Convert latent risk into probability of non-compliance
        # -------------------------------------------------------------
        noncompliance_probability = sigmoid(
            5.0 * (risk_score - 0.55)
        )

        # -------------------------------------------------------------
        # Probabilistic compliance assignment
        # -------------------------------------------------------------
        non_compliant = int(
            np.random.random()
            < noncompliance_probability
        )

        compliant = (
            1 - non_compliant
        )

        # -------------------------------------------------------------
        # Build final record
        # -------------------------------------------------------------
        records.append(
            {
                "record_id": (
                    f"{provider}-synth-{i+1:05d}"
                ),
                "provider": provider,
                "resource_type": np.random.choice(
                    RESOURCE_TYPES[provider]
                ),
                "region": np.random.choice(
                    REGIONS[provider]
                ),
                "iam_score": iam_score,
                "encryption_status": encryption_status,
                "network_exposure": network_exposure,
                "logging_status": logging_status,
                "compliance_status": compliant,
                "data_source": "synthetic_calibrated",
            }
        )

    return records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    real_df = pd.read_csv(
        "real_evidence_dataset.csv"
    )

    # Approximately 1000 synthetic records per provider
    all_records = []

    for provider, n_records in [
        ("aws", 1000),
        ("azure", 1000),
        ("gcp", 1000),
    ]:

        all_records.extend(
            generate_provider_records(
                provider,
                n_records,
            )
        )

    synth_df = pd.DataFrame(
        all_records
    )

    # -----------------------------------------------------------------
    # Keep real evidence separate from ML training
    # -----------------------------------------------------------------
    
    synth_df.to_csv(
        "gdpr_compliance_training_dataset.csv",
        index=False,
    )

    real_df.to_csv(
        "real_evidence_dataset.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------
    print(
        "=== Training dataset "
        "(synthetic, evidence-calibrated) ==="
    )

    print(
        "Total records:",
        len(synth_df),
    )

    print(
        synth_df.groupby(
            "provider"
        ).size()
    )

    print()

    print(
        "Missing values:",
        synth_df[
            [
                "iam_score",
                "encryption_status",
                "network_exposure",
                "logging_status",
            ]
        ]
        .isna()
        .sum()
        .sum(),
    )

    print()

    print(
        "=== Compliance status distribution ==="
    )

    counts = (
        synth_df[
            "compliance_status"
        ]
        .value_counts()
        .sort_index()
    )

    proportions = (
        synth_df[
            "compliance_status"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print(
        "Counts:"
    )

    print(
        counts
    )

    print()

    print(
        "Proportions:"
    )

    print(
        proportions
    )

    print()

    print(
        "0 = Non-compliant"
    )

    print(
        "1 = Compliant"
    )

    print()

    print(
        "=== Synthetic fail-rate sanity check: AWS ==="
    )

    aws_synth = synth_df[
        synth_df["provider"]
        == "aws"
    ]

    print(
        "AWS IAM score mean:",
        round(
            aws_synth[
                "iam_score"
            ].mean(),
            3,
        ),
    )

    print(
        "AWS encryption failure rate:",
        round(
            (
                aws_synth[
                    "encryption_status"
                ]
                == 0
            ).mean(),
            3,
        ),
    )

    print(
        "AWS network exposure rate:",
        round(
            (
                aws_synth[
                    "network_exposure"
                ]
                > 0
            ).mean(),
            3,
        ),
    )

    print(
        "AWS logging failure rate:",
        round(
            (
                aws_synth[
                    "logging_status"
                ]
                == 0
            ).mean(),
            3,
        ),
    )

    print()

    print(
        "Saved:"
    )

    print(
        "gdpr_compliance_training_dataset.csv"
    )

    print(
        "real_evidence_dataset.csv"
    )


if __name__ == "__main__":
    main()

