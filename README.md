# AI-Driven Detection of GDPR Article 32 Non-Compliance in Multi-Cloud Environments

**MSc Cyber Security — Applied Research Project**  
**Rakhimol Raju | Dublin Business School | 2026**

A Python-based research pipeline and interactive Streamlit dashboard for analysing cloud-security misconfigurations and predicting potential GDPR Article 32 non-compliance across **AWS, Microsoft Azure and Google Cloud Platform (GCP)**.

> **Important:** This is a research prototype. Its predictions are technical risk indicators based on the study's operational definition of Article 32 compliance. They are **not legal determinations or GDPR compliance certification**.

## Project overview

This project examines four measurable cloud-security factors:

- **Identity and Access Management (IAM):** access-control weaknesses and excessive permissions.
- **Encryption:** protection of data through encryption controls.
- **Logging:** audit logging and monitoring controls.
- **Network exposure:** externally reachable resources and network configurations.

Real security evidence from **Prowler** scans across AWS, Azure and GCP and IAM-related signals from **AWS CloudTrail** is used to calibrate a synthetic modelling dataset. Statistical analysis examines associations between the four security factors and the study's Article 32 compliance label. Three machine-learning classifiers are then compared with a deterministic rule-based baseline.

## Research methodology

The study follows a **positivist philosophy**, **deductive approach** and **quantitative methodology**. It combines cross-sectional association analysis with comparative machine-learning evaluation.

The workflow is:

**CloudTrail and Prowler evidence → dataset construction → real/synthetic validation → preprocessing → statistical analysis → model training and evaluation → explainability → bootstrap confidence intervals → Streamlit dashboard**

## Dataset

| Stage | Records |
|---|---:|
| Initially generated dataset | 3,000 (1,000 per cloud provider) |
| Exact duplicates removed | 203 |
| Final analytical dataset | 2,797 |
| Stratified training set | 2,237 |
| Held-out test set | 560 |

The synthetic dataset is **calibrated using available real evidence**, rather than being presented as a complete real-world organisational dataset. Real-versus-synthetic validation compared 12 provider/control failure proportions. Following Holm correction, 11 comparisons showed no statistically significant difference; **GCP logging** remained a significant discrepancy. A non-significant result is not proof that the datasets are equivalent.

For modelling, **class `1` represents non-compliance** and **class `0` represents compliance**. The original dataset's `compliance_status` column uses the opposite encoding: `0` = non-compliant and `1` = compliant. Check the convention when using individual scripts or interpreting plots.

## Repository scripts

The main scripts are listed in their intended pipeline order. Filenames retain their original repository spelling.

| Script | Purpose |
|---|---|
| `1_extract_cloudtrail_iam_signal.py` | Extract IAM-related signals from CloudTrail logs and calculate identity-level IAM scores. |
| `2_build_real_dataset.py` | Categorise Prowler findings into IAM, encryption, logging and network exposure; build resource-level real-evidence records. |
| `3_build_final_dataset.py` | Generate the evidence-calibrated modelling dataset across AWS, Azure and GCP. |
| `11_validate_real_vs_synthetic.py` | Compare real and synthetic provider/control failure rates using appropriate proportion tests and Holm correction. |
| `4_step2_preprocessing.py` | Remove duplicates, support manual label review, split the data and perform training-only preprocessing. |
| `5_step3_statistical_analysis.py` | Test security-factor associations using Chi-square/Fisher tests and Cramér's V, with multiple-testing correction. |
| `6_step4_model_training.py` | Tune, train and evaluate Logistic Regression, Random Forest, XGBoost and the rule-based baseline; run McNemar comparisons and explainability analysis. |
| `7_staistical_visualizaton.py` | Generate statistical-analysis figures. |
| `8_Model_visualization.py` | Generate confusion matrices, ROC curves and model/explainability visualisations. |
| `9_step5_bootstrap_evaluation(1).py` | Estimate 95% confidence intervals using 1,000 stratified bootstrap resamples. |
| `10_dashboard.py` | Run the interactive Streamlit prediction dashboard. |
| `12_count.py` | Display provider counts from the training dataset. |
| `13_dataset_percentage.py` | Calculate failure counts and percentages from real evidence. |

Other experimental or earlier-version scripts may also be present in the repository; the table above describes the principal final workflow.

## Technologies

- **Python**, pandas, NumPy and SciPy
- **scikit-learn** and **XGBoost**
- **SHAP** for model explainability
- **statsmodels** for statistical testing
- **Matplotlib** and **Plotly** for visualisation
- **Streamlit** for the interactive dashboard
- **Joblib** for saving and loading trained models
- **Faker** for synthetic record generation

## Getting started


### 1. Create and activate a virtual environment

On Windows (PowerShell):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

If the repository contains a tested `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

Otherwise, the principal packages used by the scripts can be installed with:

```bash
pip install pandas numpy scipy scikit-learn xgboost shap statsmodels matplotlib plotly streamlit joblib faker
```

The command above is a package list, **not a tested, version-pinned environment**. Dependency versions may need adjustment for your Python version.

### 4. Prepare the input files

The extraction and real-evidence scripts require their respective **CloudTrail** and **Prowler** input files. Their paths and expected formats are defined in the scripts. The CloudTrail extractor accepts an input directory through `--input-dir`; the Prowler processing script reads provider-specific scan CSV files.

If your repository already includes the processed CSVs and saved `.joblib` models, you can use those files instead of regenerating the entire pipeline. Do not publish private cloud logs, credentials, personally identifying data or other sensitive scan details.

### 5. Run the pipeline

Run the scripts from the directory containing the project files, in the following order, once the required input data is available:

```bash
python 1_extract_cloudtrail_iam_signal.py --input-dir <CLOUDTRAIL_FOLDER>
python 2_build_real_dataset.py
python 3_build_final_dataset.py
python 11_validate_real_vs_synthetic.py
python 4_step2_preprocessing.py
python 5_step3_statistical_analysis.py
python 6_step4_model_training.py
python 7_staistical_visualizaton.py
python 8_Model_visualization.py
python "9_step5_bootstrap_evaluation(1).py"
```

**Notes:** Replace `<CLOUDTRAIL_FOLDER>` with the actual folder containing your CloudTrail logs. Review the input paths in each script before running. The preprocessing script includes a manual label-review stage; complete that review if you want to reproduce the reported label-agreement result. Some scripts depend on outputs from earlier steps and are not designed to run independently.

### 6. Launch the Streamlit dashboard

If the saved model files are available in the location expected by `10_dashboard.py`, run:

```bash
streamlit run 10_dashboard.py
```

Streamlit normally opens the application in your browser. Otherwise, open the local URL displayed in the terminal, usually `http://localhost:8501`.

The dashboard expects the saved models:

```text
model_lr.joblib
model_rf.joblib
model_xgb.joblib
```

## Model development and evaluation

Three classifiers are tuned with **Grid Search** and **five-fold stratified cross-validation**, using F1-score as the tuning metric:

1. Logistic Regression
2. Random Forest
3. XGBoost

A deterministic **rule-based baseline** flags a configuration when any of the defined safeguards fails. All approaches are evaluated on the same held-out test set.

### Reported held-out results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 77.14% | 78.49% | 91.95% | 0.847 | 0.821 |
| Random Forest | 75.00% | 79.24% | 86.23% | 0.826 | 0.798 |
| XGBoost | 76.61% | 76.24% | 95.84% | 0.849 | 0.815 |
| Rule-based baseline | 69.82% | 69.57% | 99.74% | 0.820 | N/A |

The rule-based baseline has no ROC-AUC value in the reported comparison. McNemar's test compares paired classification outcomes on the same test records; bootstrap evaluation uses **1,000 stratified resamples** to estimate 95% confidence intervals for model metrics.

### Statistical analysis and explainability

The study uses Chi-square or Fisher's Exact Test, as appropriate, and **Cramér's V** to examine association strength. **Random Forest feature importance** and **XGBoost SHAP** provide model-specific explanations. IAM had the strongest average association and the highest importance in both reported model-specific analyses. These findings indicate **association and predictive importance, not causation**.

## Dashboard features

The Streamlit dashboard allows a user to:

- Select AWS, Azure or GCP.
- Enter IAM score, encryption status, logging status and network exposure.
- Select a trained model and view its predicted compliance class, non-compliance probability and risk level.
- Compare outputs from the three trained models and the rule-based baseline.
- Compare the same security configuration across cloud providers.
- Review security observations and previously tested scenarios within the application.

Provider comparisons hold the entered security factors constant and vary the provider input; they do **not** establish that one provider is inherently more or less GDPR-compliant.

## Main generated files

Examples of outputs produced by the pipeline include:

```text
cloudtrail_iam_summary.csv
real_evidence_dataset.csv
gdpr_compliance_training_dataset.csv
real_vs_synthetic_validation.csv
real_vs_synthetic_validation_chart.png
train_set.csv
test_set.csv
statistical_analysis_results.csv
model_lr.joblib
model_rf.joblib
model_xgb.joblib
hyperparameter_tuning_results.csv
model_evaluation_results.csv
rf_feature_importance.csv
xgboost_shap_importance.csv
mcnemar_results.csv
bootstrap_confidence_intervals.csv
bootstrap_all_iterations.csv
bootstrap_summary_table.csv
```

Statistical and model visualisation scripts also save figures such as confusion matrices, ROC curves and feature-importance charts.

## Limitations and future work

- The modelling dataset is partly synthetic and is calibrated from limited real evidence; it is not a representative sample of all production cloud environments.
- Real-versus-synthetic validation assesses individual provider/control failure proportions, not every joint relationship in the generated data.
- The GCP logging validation comparison showed a statistically significant discrepancy.
- The selected four factors are measurable technical indicators related to Article 32; they do not cover every legal, organisational or contextual requirement.
- Future work should validate the approach with larger, diverse real organisational datasets and evaluate its performance in operational cloud environments.

## Academic use and disclaimer

This repository accompanies an MSc Cyber Security Applied Research Project at Dublin Business School. It is intended for research, education and demonstration. The dashboard provides predictive risk indicators only and should not be used as a substitute for a full security assessment or professional legal advice.

**Author:** Rakhimol Raju  
**Programme:** MSc Cyber Security, Dublin Business School  
**Year:** 2026
# -GDPR-AI-Compliance-Analytics-Engine--dissertation
