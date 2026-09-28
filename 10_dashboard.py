import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GDPR Article 32 Compliance Prediction",
    page_icon="🔐",
    layout="wide",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    div[data-testid="stMetric"] {
        background-color: #f7f9fc;
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        padding: 14px 18px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        color: white;
    }

    .risk-low {
        background-color: #2e9e5b;
    }

    .risk-medium {
        background-color: #e0a72c;
    }

    .risk-high {
        background-color: #d64545;
    }

    .observation-card {
        background-color: #fff6f6;
        border-left: 4px solid #d64545;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 6px;
    }

    .success-card {
        background-color: #f2fbf5;
        border-left: 4px solid #2e9e5b;
        padding: 10px 14px;
        border-radius: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FINAL MODELS
# ============================================================

@st.cache_resource
def load_models():

    return {
        "Logistic Regression":
            joblib.load("model_lr.joblib"),

        "Random Forest":
            joblib.load("model_rf.joblib"),

        "XGBoost":
            joblib.load("model_xgb.joblib"),
    }


@st.cache_data
def load_model_stats():

    if os.path.exists("model_evaluation_results.csv"):

        return pd.read_csv(
            "model_evaluation_results.csv"
        )

    return None


models = load_models()
model_stats = load_model_stats()


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
    "iam_score",
    "encryption_status",
    "logging_status",
    "network_exposure_0",
    "network_exposure_1",
    "network_exposure_2",
    "provider_aws",
    "provider_azure",
    "provider_gcp",
]


# ============================================================
# RULE-BASED BASELINE
# ============================================================

def rule_based_prediction(
    iam_score,
    encryption_status,
    logging_status,
    network_exposure
):

    iam_fail = (
        iam_score > 0.5
    )

    encryption_fail = (
        encryption_status == 0
    )

    logging_fail = (
        logging_status == 0
    )

    network_fail = (
        network_exposure in [1, 2]
    )

    non_compliant = (
        iam_fail
        or encryption_fail
        or logging_fail
        or network_fail
    )

    return int(
        non_compliant
    )


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def create_input_dataframe(
    iam_score,
    encryption_status,
    logging_status,
    network_exposure,
    provider
):

    network_0 = (
        1 if network_exposure == 0 else 0
    )

    network_1 = (
        1 if network_exposure == 1 else 0
    )

    network_2 = (
        1 if network_exposure == 2 else 0
    )

    provider_aws = (
        1 if provider == "AWS" else 0
    )

    provider_azure = (
        1 if provider == "Azure" else 0
    )

    provider_gcp = (
        1 if provider == "GCP" else 0
    )

    input_data = pd.DataFrame(
        [
            {
                "iam_score":
                    iam_score,

                "encryption_status":
                    encryption_status,

                "logging_status":
                    logging_status,

                "network_exposure_0":
                    network_0,

                "network_exposure_1":
                    network_1,

                "network_exposure_2":
                    network_2,

                "provider_aws":
                    provider_aws,

                "provider_azure":
                    provider_azure,

                "provider_gcp":
                    provider_gcp,
            }
        ]
    )

    input_data = (
        input_data[
            FEATURE_COLUMNS
        ]
    )

    return input_data


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(
    probability
):

    if probability < 0.34:

        return (
            "Low",
            "risk-low"
        )

    elif probability < 0.67:

        return (
            "Medium",
            "risk-medium"
        )

    else:

        return (
            "High",
            "risk-high"
        )


# ============================================================
# PROBABILITY GAUGE
# ============================================================

def make_gauge(
    probability
):

    level, _ = (
        risk_level(
            probability
        )
    )

    color_mapping = {
        "Low":
            "#2e9e5b",

        "Medium":
            "#e0a72c",

        "High":
            "#d64545",
    }

    color = (
        color_mapping[
            level
        ]
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=(
                probability
                * 100
            ),

            number={
                "suffix": "%",
                "font": {
                    "size": 40
                }
            },

            gauge={
                "axis": {
                    "range": [
                        0,
                        100
                    ]
                },

                "bar": {
                    "color":
                        color
                },

                "steps": [
                    {
                        "range": [0, 34],
                        "color": "#e9f7ee"
                    },

                    {
                        "range": [34, 67],
                        "color": "#fdf3e0"
                    },

                    {
                        "range": [67, 100],
                        "color": "#fbeaea"
                    },
                ],

                "threshold": {
                    "line": {
                        "color":
                            "black",

                        "width":
                            3
                    },

                    "thickness":
                        0.8,

                    "value":
                        probability
                        * 100,
                },
            },

            title={
                "text":
                    "Non-Compliance Probability"
            },
        )
    )

    fig.update_layout(
        height=280,

        margin=dict(
            t=50,
            b=10,
            l=30,
            r=30
        )
    )

    return fig


# ============================================================
# SECURITY PROFILE RADAR
# ============================================================

def make_radar(
    iam_score,
    encryption_status,
    logging_status,
    network_exposure
):

    categories = [
        "IAM",
        "Encryption",
        "Logging",
        "Network",
    ]

    current = [
        1 - iam_score,
        encryption_status,
        logging_status,
        1 - (
            network_exposure / 2
        ),
    ]

    ideal = [
        1,
        1,
        1,
        1,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=(
                ideal
                + [ideal[0]]
            ),

            theta=(
                categories
                + [categories[0]]
            ),

            fill="toself",

            name="Reference profile",

            line=dict(
                color="#2e9e5b",
                dash="dot"
            ),

            opacity=0.35,
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=(
                current
                + [current[0]]
            ),

            theta=(
                categories
                + [categories[0]]
            ),

            fill="toself",

            name="Selected configuration",

            line=dict(
                color="#3b6fd6"
            ),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),

        showlegend=True,

        height=420,

        margin=dict(
            t=40,
            b=60,
            l=80,
            r=80
        ),
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔐 About"
    )

    st.write(
        """
        This dashboard applies machine-learning models
        trained on cloud-security configuration data to
        estimate whether a resource is likely to be
        non-compliant with GDPR Article 32.
        """
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Model Performance"
    )

    if model_stats is not None:

        metric_map = {
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

        available = [
            column
            for column
            in metric_map
            if column
            in model_stats.columns
        ]

        display_stats = (
            model_stats[
                [
                    "model"
                ]
                + available
            ]
            .rename(
                columns=metric_map
            )
        )

        display_stats = (
            display_stats.rename(
                columns={
                    "model":
                        "Model"
                }
            )
        )

        st.dataframe(
            display_stats
            .set_index(
                "Model"
            )
            .round(3),

            use_container_width=True,
        )

    else:

        st.caption(
            """
            Model performance results are not available.
            Run the model evaluation stage first.
            """
        )

    st.markdown(
        "---"
    )

    st.caption(
        """
        ML target convention:
        **1 = Non-compliant, 0 = Compliant**

        Predictions reflect patterns learned from the
        study dataset and do not constitute a legal
        determination of GDPR compliance.
        """
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "GDPR Article 32 Non-Compliance Prediction"
)

st.write(
    """
    Enter a cloud-security configuration to estimate
    whether it is likely to be compliant or non-compliant
    with GDPR Article 32.
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "Cloud Security Configuration"
)

col1, col2 = (
    st.columns(2)
)


with col1:

    provider = (
        st.selectbox(
            "Cloud Provider",
            [
                "AWS",
                "Azure",
                "GCP"
            ]
        )
    )

    iam_score = (
        st.slider(
            "IAM Misconfiguration Score (0–1)",

            min_value=0.0,

            max_value=1.0,

            value=0.50,

            step=0.01,

            help=(
                "Higher values indicate greater "
                "IAM misconfiguration."
            )
        )
    )

    encryption_option = (
        st.selectbox(
            "Encryption Status",
            [
                "Enabled",
                "Disabled"
            ]
        )
    )


with col2:

    logging_option = (
        st.selectbox(
            "Logging Status",
            [
                "Enabled",
                "Disabled"
            ]
        )
    )

    network_option = (
        st.selectbox(
            "Network Exposure",
            [
                "No exposure",
                "Moderate exposure",
                "High exposure"
            ]
        )
    )

    selected_model_name = (
        st.selectbox(
            "Prediction Model",
            [
                "XGBoost",
                "Logistic Regression",
                "Random Forest"
            ]
        )
    )


# ============================================================
# CONVERT INPUT VALUES
# ============================================================

encryption_status = (
    1
    if encryption_option
    == "Enabled"
    else 0
)

logging_status = (
    1
    if logging_option
    == "Enabled"
    else 0
)

network_mapping = {
    "No exposure":
        0,

    "Moderate exposure":
        1,

    "High exposure":
        2,
}

network_exposure = (
    network_mapping[
        network_option
    ]
)


# ============================================================
# CREATE INPUT
# ============================================================

input_df = (
    create_input_dataframe(
        iam_score,
        encryption_status,
        logging_status,
        network_exposure,
        provider
    )
)


# ============================================================
# SESSION HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_clicked = (
    st.button(
        "Predict Article 32 Status",
        type="primary",
        use_container_width=True
    )
)


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_clicked:

    selected_model = (
        models[
            selected_model_name
        ]
    )

    prediction = int(
        selected_model.predict(
            input_df
        )[0]
    )

    probability = float(
        selected_model.predict_proba(
            input_df
        )[0][1]
    )

    baseline_prediction = (
        rule_based_prediction(
            iam_score,
            encryption_status,
            logging_status,
            network_exposure
        )
    )

    level, level_class = (
        risk_level(
            probability
        )
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    st.session_state.history.append(
        {
            "Provider":
                provider,

            "IAM":
                round(
                    iam_score,
                    2
                ),

            "Encryption":
                encryption_option,

            "Logging":
                logging_option,

            "Network":
                network_option,

            "Model":
                selected_model_name,

            "Prediction":
                (
                    "Non-compliant"
                    if prediction == 1
                    else "Compliant"
                ),

            "Probability":
                f"{probability * 100:.1f}%",
        }
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader(
        "Prediction Result"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        st.metric(
            "Predicted Status",

            (
                "NON-COMPLIANT"
                if prediction == 1
                else "COMPLIANT"
            )
        )

    with result_col2:

        st.metric(
            "Non-Compliance Probability",

            f"{probability * 100:.1f}%"
        )

    with result_col3:

        st.metric(
            "Model Used",
            selected_model_name
        )

    st.markdown(
        (
            f'<span class="risk-badge '
            f'{level_class}">'
            f'Risk level: {level}'
            f'</span>'
        ),

        unsafe_allow_html=True,
    )

    st.write("")


    # ========================================================
    # GAUGE + RADAR
    # ========================================================

    gauge_col, radar_col = (
        st.columns(2)
    )

    with gauge_col:

        st.plotly_chart(
            make_gauge(
                probability
            ),

            use_container_width=True
        )

    with radar_col:

        st.plotly_chart(
            make_radar(
                iam_score,
                encryption_status,
                logging_status,
                network_exposure
            ),

            use_container_width=True
        )

        st.caption(
            """
            A larger selected-configuration area closer
            to the reference profile indicates stronger
            security-control configuration.
            """
        )


    # ========================================================
    # CONFIGURATION SUMMARY
    # ========================================================

    st.write(
        "### Configuration Summary"
    )

    summary_data = pd.DataFrame(
        {
            "Security Factor": [
                "Cloud Provider",
                "IAM Misconfiguration Score",
                "Encryption",
                "Logging",
                "Network Exposure",
            ],

            "Configuration": [
                provider,
                f"{iam_score:.2f}",
                encryption_option,
                logging_option,
                network_option,
            ],
        }
    )

    st.dataframe(
        summary_data,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.write(
        "### Prediction Across All Models"
    )

    comparison_results = []

    for (
        model_name,
        model
    ) in models.items():

        model_prediction = int(
            model.predict(
                input_df
            )[0]
        )

        model_probability = float(
            model.predict_proba(
                input_df
            )[0][1]
        )

        comparison_results.append(
            {
                "Model":
                    model_name,

                "Prediction":
                    (
                        "Non-compliant"
                        if model_prediction == 1
                        else "Compliant"
                    ),

                "Non-Compliance Probability":
                    f"{model_probability * 100:.2f}%",
            }
        )


    comparison_results.append(
        {
            "Model":
                "Rule-Based Baseline",

            "Prediction":
                (
                    "Non-compliant"
                    if baseline_prediction == 1
                    else "Compliant"
                ),

            "Non-Compliance Probability":
                "N/A",
        }
    )


    comparison_df = (
        pd.DataFrame(
            comparison_results
        )
    )

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True
    )


    unique_predictions = {
        row["Prediction"]
        for row in comparison_results
    }


    if len(
        unique_predictions
    ) == 1:

        st.success(
            """
            All models and the rule-based baseline
            agree on this classification.
            """
        )

    else:

        st.info(
            """
            The models do not all agree on this
            configuration. Interpret the prediction
            with additional caution.
            """
        )


    # ========================================================
    # PROVIDER COMPARISON
    # ========================================================

    st.write(
        "### Same Configuration Across Cloud Providers"
    )

    st.caption(
        """
        IAM, encryption, logging and network exposure are
        held constant. Only the cloud provider is changed.
        This shows whether the selected model's prediction
        varies between AWS, Azure and GCP.
        """
    )


    provider_comparison = []


    for comparison_provider in [
        "AWS",
        "Azure",
        "GCP"
    ]:

        provider_input = (
            create_input_dataframe(
                iam_score,
                encryption_status,
                logging_status,
                network_exposure,
                comparison_provider
            )
        )

        provider_prediction = int(
            selected_model.predict(
                provider_input
            )[0]
        )

        provider_probability = float(
            selected_model.predict_proba(
                provider_input
            )[0][1]
        )

        provider_comparison.append(
            {
                "Cloud Provider":
                    comparison_provider,

                "Prediction":
                    (
                        "Non-compliant"
                        if provider_prediction == 1
                        else "Compliant"
                    ),

                "Non-Compliance Probability":
                    provider_probability,
            }
        )


    provider_comparison_df = (
        pd.DataFrame(
            provider_comparison
        )
    )


    # --------------------------------------------------------
    # PROVIDER TABLE
    # --------------------------------------------------------

    provider_display_df = (
        provider_comparison_df.copy()
    )

    provider_display_df[
        "Non-Compliance Probability"
    ] = (
        provider_display_df[
            "Non-Compliance Probability"
        ]
        .apply(
            lambda x:
                f"{x * 100:.2f}%"
        )
    )


    st.dataframe(
        provider_display_df,
        hide_index=True,
        use_container_width=True
    )


    # --------------------------------------------------------
    # PROVIDER CHART
    # --------------------------------------------------------

    provider_chart = (
        go.Figure()
    )

    provider_chart.add_trace(
        go.Bar(
            x=(
                provider_comparison_df[
                    "Cloud Provider"
                ]
            ),

            y=(
                provider_comparison_df[
                    "Non-Compliance Probability"
                ]
                * 100
            ),

            text=(
                provider_comparison_df[
                    "Non-Compliance Probability"
                ]
                * 100
            ).round(1),

            texttemplate=(
                "%{text}%"
            ),

            textposition=(
                "outside"
            ),
        )
    )


    provider_chart.update_layout(
        title=(
            f"Provider Comparison Using "
            f"{selected_model_name}"
        ),

        xaxis_title=(
            "Cloud Provider"
        ),

        yaxis_title=(
            "Non-Compliance Probability (%)"
        ),

        yaxis=dict(
            range=[
                0,
                100
            ]
        ),

        height=400,

        margin=dict(
            t=60,
            b=40,
            l=50,
            r=30
        )
    )


    st.plotly_chart(
        provider_chart,
        use_container_width=True
    )


    # --------------------------------------------------------
    # PROVIDER VARIATION INTERPRETATION
    # --------------------------------------------------------

    probabilities = (
        provider_comparison_df[
            "Non-Compliance Probability"
        ]
    )

    difference = (
        probabilities.max()
        - probabilities.min()
    )

    highest_row = (
        provider_comparison_df.loc[
            probabilities.idxmax()
        ]
    )

    lowest_row = (
        provider_comparison_df.loc[
            probabilities.idxmin()
        ]
    )


    if difference < 0.05:

        st.info(
            """
            The selected model produces very similar
            non-compliance probabilities across AWS,
            Azure and GCP for this configuration.
            """
        )

    else:

        st.info(
            f"""
            For the same security configuration,
            **{highest_row['Cloud Provider']}** produced
            the highest non-compliance probability
            (**{highest_row['Non-Compliance Probability'] * 100:.1f}%**),
            while **{lowest_row['Cloud Provider']}**
            produced the lowest
            (**{lowest_row['Non-Compliance Probability'] * 100:.1f}%**).
            """
        )


    # ========================================================
    # SECURITY OBSERVATIONS
    # ========================================================

    st.write(
        "### Security Observations"
    )

    observations = []


    if iam_score > 0.5:

        observations.append(
            "IAM misconfiguration score is elevated."
        )


    if encryption_status == 0:

        observations.append(
            "Encryption is disabled."
        )


    if logging_status == 0:

        observations.append(
            "Security logging is disabled."
        )


    if network_exposure == 1:

        observations.append(
            "Moderate network exposure is present."
        )

    elif network_exposure == 2:

        observations.append(
            "High network exposure is present."
        )


    if not observations:

        st.markdown(
            (
                '<div class="success-card">'
                '✅ No configured safeguard failure '
                'was identified from the selected '
                'input values.'
                '</div>'
            ),

            unsafe_allow_html=True,
        )

    else:

        for observation in observations:

            st.markdown(
                (
                    '<div class="observation-card">'
                    f'⚠️ {observation}'
                    '</div>'
                ),

                unsafe_allow_html=True,
            )


# ============================================================
# SCENARIO HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    with st.expander(
        (
            f"📋 Scenario History "
            f"({len(st.session_state.history)})"
        ),

        expanded=False
    ):

        history_df = (
            pd.DataFrame(
                st.session_state.history
            )
        )

        st.dataframe(
            history_df,
            hide_index=True,
            use_container_width=True
        )

        if st.button(
            "Clear History"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    This dashboard is a research prototype. Predictions
    represent patterns learned from the study dataset and
    should not be interpreted as a legal determination of
    GDPR compliance.
    """
)