# Customer Churn Prediction & Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

An enterprise-ready machine learning, relational SQL, and business analytics solution designed to predict subscription churn, prioritize high-risk customer accounts, and generate transparent, rule-based retention actions.

---

## 📌 Executive Summary

Customer churn presents an **asymmetric business cost** in telecommunications: failing to identify an at-risk customer who subsequently cancels results in the permanent loss of recurring revenue and customer lifetime value (CLTV), whereas outreach to a customer falsely flagged as at-risk carries only marginal communication costs.

This project delivers an end-to-end analytics and machine learning pipeline covering:
1. **Data Cleaning & Leakage Prevention:** Rigorous resolution of data types, domain-justified imputation of zero-tenure accounts, and elimination of retrospective target proxies.
2. **Exploratory Data Analysis (EDA):** Quantified churn hazards across contract agreements, billing tiers, payment channels, and tenure cohorts.
3. **Leak-Free Machine Learning Pipelines:** Scikit-learn ColumnTransformer architecture encapsulating continuous scaling and categorical encoding fitted strictly on training data.
4. **Comparative Modeling & Cost-Sensitive Selection:** Empirical comparison between balanced Logistic Regression and Random Forest ensembles, selecting Logistic Regression for superior churn recall (+13.6% recall advantage; 51 fewer missed churners).
5. **Operational Threshold Optimization:** Systematic threshold tuning (0.30–0.70) establishing **0.60** as the operational decision boundary, maximizing $F_1$-score ($0.6284$) and reducing false alarms by **24.6%**.
6. **Analytical Risk Segmentation:** Monotonically validated customer tiering (Low Risk: 4.9% churn, Medium Risk: 22.8% churn, High Risk: 58.1% churn).
7. **Transparent Retention Heuristics:** Non-causal business rules translating customer friction points into targeted customer success workflows.
8. **Relational SQL Layer & BI Datasets:** SQLite database storing scored portfolios with 10 production analytical queries and dimensional tables for Power BI.
9. **Interactive Streamlit Web Interface:** Real-time web application enabling customer scoring, risk tiering, and retention action generation.

---

## 🏗️ System Architecture

```text
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END SYSTEM WORKFLOW                                      |
+----------------------------------------------------------------------------------------------------+
| 1. RAW INGESTION & DATA CLEANING (src/data_cleaning.py)                                            |
|    - Source: data/raw/Telco_customer_churn.xlsx (7,043 rows x 33 columns)                          |
|    - Coerced Total Charges to numeric; imputed 11 brand-new accounts (Tenure=0) with $0.00         |
|    - Verified 0 duplicates, valid binary target, and non-negative billing values                   |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                                   ▼
+----------------------------------------------------------------------------------------------------+
| 2. FEATURE TAXONOMY & LEAKAGE PREVENTION (src/feature_selection.py)                               |
|    - Removed 1 ID (CustomerID), 3 zero-variance constants (Count, Country, State)                  |
|    - Removed 5 high-cardinality/spatial noise features (City, Zip Code, Lat Long, Lat, Long)        |
|    - Removed 4 target leakage columns (Churn Label, Churn Score, CLTV, Churn Reason)               |
|    - Isolated 19 approved observable model features (3 numerical, 16 categorical)                  |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                                   ▼
+----------------------------------------------------------------------------------------------------+
| 3. PREPROCESSING & DATA SPLITTING (src/preprocess.py)                                              |
|    - Stratified 80/20 train/test split (random_state=42; 5,634 Train / 1,409 Test)                 |
|    - ColumnTransformer: StandardScaler (numeric) + OneHotEncoder(handle_unknown='ignore')          |
|    - Preprocessor fitted strictly on training data inside scikit-learn Pipelines                   |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                                   ▼
+----------------------------------------------------------------------------------------------------+
| 4. MODEL TRAINING & COMPARATIVE SELECTION (src/train_logistic.py, src/compare_models.py)           |
|    - Fitted Logistic Regression (class_weight='balanced') and Random Forest (200 estimators)       |
|    - Logistic Regression selected: 78.07% Recall vs. 64.44% RF (+51 churners caught, ROC-AUC 0.85) |
|    - Threshold optimized at 0.60: Peak F1=0.6284, reducing False Positives from 280 to 211         |
+----------------------------------------------------------------------------------------------------+
                                                   │
                                                   ▼
+----------------------------------------------------------------------------------------------------+
| 5. RISK SEGMENTATION & HEURISTIC RETENTION (src/risk_segmentation.py, src/retention_strategy.py)   |
|    - Low Risk (<0.30): 4.9% churn | Med Risk (0.30-0.60): 22.8% churn | High Risk (>=0.60): 58.1%   |
|    - Data-driven Q3 billing cutoff ($89.85); composite retention actions; Priority Rank tracking   |
+----------------------------------------------------------------------------------------------------+
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
+---------------------------------------------------+ +----------------------------------------------+
| 6. RELATIONAL SQL ANALYTICS (sql/churn_analysis)  | | 7. STREAMLIT APP & BI DATA (app.py, powerbi) |
|    - SQLite database: data/processed/customer_    | |    - Interactive real-time scoring app       |
|      churn.db (table: customer_churn, N=7,043)    | |    - Presets: High-Risk, Moderate, Loyal     |
|    - Queries A-J: KPIs, contract/tenure breakdowns| |    - Observable drivers & retention alerts   |
|    - Verified 0.00% discrepancy with Python EDA   | |    - Power BI dimensions & KPI summaries     |
+---------------------------------------------------+ +----------------------------------------------+
```

---

## 📊 Key Machine Learning & Modeling Results

### 1. Test Set Model Comparison ($N = 1,409$, 374 Actual Churners)

Models were evaluated under an identical stratified 80/20 test split (`random_state=42`).

| Evaluation Metric | Logistic Regression (Baseline $t=0.50$) | Random Forest (Baseline $t=0.50$) | Logistic Regression (Optimized $t=0.60$) | Business Impact / Rationale |
|---|:---:|:---:|:---:|---|
| **Accuracy** | 74.31% | 77.36% | **77.50%** | Higher overall correct classifications |
| **Precision (Churn Class 1)** | 51.05% | 56.44% | **55.95%** | +4.9% precision improvement at $t=0.60$ |
| **Recall (Churn Class 1)** | **78.07%** | 64.44% | **71.66%** | **Catches 268 churners** (RF catches only 241) |
| **F1-Score (Class 1)** | 0.6173 | 0.6017 | **0.6284** | **Peak $F_1$-score across all candidate thresholds** |
| **ROC-AUC** | **0.8489** | 0.8333 | **0.8489** | Strong global discrimination capability |
| **True Positives (TP)** | **292** | 241 | **268** | Catches 27 more churners than Random Forest |
| **False Negatives (FN)** | **82** | 133 | **106** | 27 fewer costly missed churners than Random Forest |
| **False Positives (FP)** | 280 | 186 | **211** | **69 false alarms eliminated** vs. baseline $t=0.50$ |
| **Detection Rate** | **78.07%** | 64.44% | **71.66%** | Majority of churners captured prior to departure |

> **Model Selection Decision:** Logistic Regression was selected over Random Forest because missing an at-risk customer (False Negative) forfeits full customer lifetime value, whereas outreach to a retained customer (False Positive) carries minimal operational cost. Logistic Regression delivers higher Recall, higher $F_1$, and superior ROC-AUC while preserving linear interpretability.

---

### 2. Analytical Risk Segmentation Validation ($N = 7,043$)

The full customer portfolio was scored using the preferred pipeline and segmented into three operational prioritization tiers:

| Risk Level | Probability Range | Customer Count | % of Base | Actual Churned | Observed Churn Rate | Avg Monthly Billing | Avg Tenure |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Low Risk** | $p < 0.30$ | 3,178 | 45.12% | 156 | **4.91%** | \$56.18 | 47.3 mos |
| **Medium Risk** | $0.30 \le p < 0.60$ | 1,507 | 21.40% | 343 | **22.76%** | \$63.41 | 27.7 mos |
| **High Risk** | $p \ge 0.60$ | 2,358 | 33.48% | 1,370 | **58.10%** | \$77.19 | 15.3 mos |
| **Total Base** | Full Spectrum | **7,043** | **100.00%** | **1,869** | **26.54%** | **\$64.76** | **32.4 mos** |

*Validation Insight:* Observed churn scales **monotonically** across risk tiers: High-Risk accounts exhibit an observed churn rate **11.8x higher** than Low-Risk accounts.

---

## 📈 Exploratory Data Analysis (EDA) Highlights

Analysis across the 7,043 customer accounts revealed pronounced structural churn drivers:
- **Contract Agreement:** Month-to-month contracts exhibit a **42.71%** churn rate, compared to **11.27%** for one-year contracts and **2.83%** for two-year contracts (**15.1x difference**).
- **Tenure Lifecycle:** Churn hazard is front-loaded; customers with $\le 12$ months tenure churn at **47.44%**, decaying monotonically to **6.61%** for customers beyond 5 years ($61+$ months).
- **Internet Infrastructure:** Fiber optic broadband subscribers churn at **41.89%**, compared to **18.96%** for DSL and **7.40%** for non-broadband accounts.
- **Payment Channel:** Manual electronic check users experience a **45.29%** churn rate, compared to **15.24%** for automatic credit card billing.
- **Support Add-Ons:** Customers lacking Technical Support (**41.64%** churn) or Online Security (**41.77%** churn) experience nearly triple the churn rate of subscribed peers (~15%).

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+ (Tested on Python 3.14)
- **Data Engineering & Analysis:** Pandas, NumPy, OpenPyXL
- **Machine Learning:** Scikit-learn (Pipelines, ColumnTransformer, LogisticRegression, RandomForestClassifier)
- **Database:** SQLite 3, SQLAlchemy
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Web Application:** Streamlit
- **Business Intelligence:** Power BI (dimensional datasets & schema prepared)

---

## 📂 Repository Structure

```text
customer-churn-prediction-retention-analytics/
├── .gitignore                                           # Clean ignore rules (excludes .venv, *.db, cache)
├── README.md                                            # Comprehensive project documentation
├── requirements.txt                                     # Pinned dependency requirements
├── app.py                                               # Interactive Streamlit web application
├── data/
│   ├── raw/
│   │   └── Telco_customer_churn.xlsx                    # Raw benchmark dataset (immutable)
│   └── processed/
│       └── telco_churn_clean.csv                        # Cleaned dataset (33 columns, 7,043 rows)
├── models/
│   ├── logistic_regression_pipeline.joblib              # Serialized production pipeline (7.7 KB)
│   └── random_forest_pipeline.joblib                    # Serialized benchmark pipeline (39.5 MB)
├── notebooks/                                           # Structured Jupyter notebooks (Phases 01 - 09)
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_threshold_analysis.ipynb
│   ├── 08_sql_business_analysis.ipynb
│   └── 09_powerbi_data_preparation.ipynb
├── outputs/
│   ├── eda_summary.txt                                  # Quantitative EDA statistical summary
│   ├── logistic_threshold_analysis.csv                  # Metrics across candidate cutoffs (0.30 - 0.70)
│   ├── model_comparison.csv                             # Formal model evaluation table
│   ├── random_forest_feature_importance.csv             # Gini feature importance rankings
│   ├── sql_*.csv                                        # 8 exported SQL business report tables
│   ├── figures/                                         # 21 high-resolution diagnostic charts
│   │   ├── 01_overall_churn_distribution.png
│   │   ├── 02_churn_by_contract.png
│   │   ├── 03_churn_by_tenure_cohort.png
│   │   ├── 04_churn_by_internet_service.png
│   │   ├── 05_churn_by_monthly_charges.png
│   │   ├── 06_churn_by_payment_method.png
│   │   ├── 07_churn_by_tech_support.png
│   │   ├── 08_churn_by_online_security.png
│   │   ├── 09_churn_by_senior_citizen.png
│   │   ├── 10_churn_by_paperless_billing.png
│   │   ├── 11_correlation_matrix.png
│   │   ├── 12_tenure_vs_monthly_charges_scatter.png
│   │   ├── logistic_confusion_matrix.png
│   │   ├── logistic_roc_curve.png
│   │   ├── logistic_threshold_tradeoff.png
│   │   ├── logistic_threshold_errors.png
│   │   ├── model_comparison_metrics.png
│   │   ├── model_roc_comparison.png
│   │   ├── random_forest_confusion_matrix.png
│   │   ├── random_forest_feature_importance.png
│   │   └── random_forest_roc_curve.png
│   └── predictions/
│       ├── customer_churn_risk_predictions.csv          # Scored predictions across all 7,043 customers
│       └── customer_retention_recommendations.csv       # Scored base with drivers & recommended actions
├── powerbi/
│   ├── customer_churn_dashboard_data.csv                # Main reporting dataset with dimensions
│   ├── customer_churn_kpi_summary.csv                   # Executive KPI summary card dataset
│   └── high_risk_customers.csv                          # Filtered priority action list (High-Risk)
├── sql/
│   └── churn_analysis.sql                               # Production SQL queries (Queries A through J)
└── src/
    ├── data_cleaning.py                                 # Ingestion & domain-defensible cleaning
    ├── feature_selection.py                             # 33-column mapping & leakage elimination
    ├── preprocess.py                                    # ColumnTransformer & stratified splitting
    ├── train_logistic.py                                # Logistic Regression pipeline training
    ├── train_random_forest.py                           # Random Forest pipeline training
    ├── compare_models.py                                # Comparative model evaluation engine
    ├── threshold_analysis.py                            # Threshold sweep & error trade-off analysis
    ├── risk_segmentation.py                             # Full base scoring & analytical risk tiering
    ├── retention_strategy.py                            # Observable risk drivers & retention actions
    ├── create_sql_database.py                           # SQLite ingestion & query execution runner
    └── prepare_powerbi_data.py                          # Dimensional data engineering for Power BI
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/customer-churn-prediction-retention-analytics.git
cd customer-churn-prediction-retention-analytics

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Streamlit Web Application
Launch the interactive web application locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Use the sidebar controls or choose from pre-built customer profiles (`High-Risk`, `Moderate-Risk`, `Low-Risk Loyal`) to explore real-time predictions, observable risk drivers, and tailored retention actions.

### 4. Reproducing the End-to-End Pipeline
Execute the modules sequentially to reproduce data cleaning, training, evaluation, database ingestion, and data preparation:
```bash
# 1. Clean raw Excel data
python src/data_cleaning.py

# 2. Inspect feature selection & leakage map
python src/feature_selection.py

# 3. Train models
python src/train_logistic.py
python src/train_random_forest.py

# 4. Compare models and analyze decision thresholds
python src/compare_models.py
python src/threshold_analysis.py

# 5. Score full customer base and generate retention recommendations
python src/risk_segmentation.py
python src/retention_strategy.py

# 6. Populate SQLite database and execute business queries
python src/create_sql_database.py

# 7. Prepare Power BI dimensional datasets
python src/prepare_powerbi_data.py
```

---

## ⚖️ Methodological & Ethical Notes

1. **Association vs. Causation:** All risk drivers and recommendations in this repository are derived from observational correlations and predictive models. They represent operational triage heuristics rather than proven causal mechanisms.
2. **Experimental Validation:** In an enterprise production setting, retention interventions (e.g., term discounts, onboarding outreach) should always be evaluated through **randomized controlled trials (A/B testing)** with holdout groups to verify treatment uplift and net incremental ROI.
3. **Model Governance:** The model utilizes demographic attributes (e.g., Senior Citizen, Gender). Production deployments should continuously audit predictions for disparate impact and adhere to responsible AI governance standards.
