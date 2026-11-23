"""
Feature Selection & Data Leakage Prevention Module for Customer Churn Analytics.

This module systematically evaluates all columns in the cleaned Telco customer dataset,
isolates the target variable ('Churn Value'), identifies and separates metadata/identifiers ('CustomerID'),
filters out constant, redundant, high-cardinality location, and data leakage columns,
and returns a clean, leak-free feature matrix (X) and target vector (y).
"""

from pathlib import Path
import pandas as pd


# Comprehensive classification mapping for all 33 columns
COLUMN_CLASSIFICATION = {
    # Target
    "Churn Value": {
        "status": "TARGET",
        "category": "Target Variable",
        "reason": "Primary ground-truth binary outcome (0 = Retained, 1 = Churned)."
    },
    
    # Identifiers
    "CustomerID": {
        "status": "REMOVE: identifier",
        "category": "Customer Identifier",
        "reason": "Unique customer ID. Carries no generalizable predictive signal and would cause overfitting; preserved separately for downstream prediction tracking."
    },
    
    # Constant columns
    "Count": {
        "status": "REMOVE: constant / no predictive variation",
        "category": "Account Metadata",
        "reason": "Constant value of 1 for all rows; zero variance, no predictive power."
    },
    "Country": {
        "status": "REMOVE: constant / no predictive variation",
        "category": "Geographic Information",
        "reason": "Constant value ('United States') for all rows; zero variance, no predictive power."
    },
    "State": {
        "status": "REMOVE: constant / no predictive variation",
        "category": "Geographic Information",
        "reason": "Constant value ('California') for all rows; zero variance, no predictive power."
    },
    
    # Redundant / High-cardinality geographic features
    "Lat Long": {
        "status": "REMOVE: redundant",
        "category": "Geographic Information",
        "reason": "String concatenation of Latitude and Longitude; redundant."
    },
    "City": {
        "status": "REMOVE: unsuitable high-cardinality/location feature",
        "category": "Geographic Information",
        "reason": "1,129 unique city categories across 7,043 rows; high cardinality creates extreme sparsity, high dimensionality, and high risk of overfitting."
    },
    "Zip Code": {
        "status": "REMOVE: unsuitable high-cardinality/location feature",
        "category": "Geographic Information",
        "reason": "1,652 unique postal codes; arbitrary discrete levels create excessive dimensionality without regional aggregation."
    },
    "Latitude": {
        "status": "REMOVE: unsuitable high-cardinality/location feature",
        "category": "Geographic Information",
        "reason": "Continuous GPS micro-coordinates that risk tree-based models memorizing spatial noise rather than customer behavior."
    },
    "Longitude": {
        "status": "REMOVE: unsuitable high-cardinality/location feature",
        "category": "Geographic Information",
        "reason": "Continuous GPS micro-coordinates that risk tree-based models memorizing spatial noise rather than customer behavior."
    },
    
    # Data leakage columns
    "Churn Label": {
        "status": "REMOVE: data leakage",
        "category": "Churn Information",
        "reason": "Direct text representation ('Yes'/'No') of the target outcome; using it would cause 100% artificial target leakage."
    },
    "Churn Score": {
        "status": "REMOVE: data leakage",
        "category": "Churn Information",
        "reason": "Pre-computed model risk score (0-100). Training on an existing model's output introduces proxy target leakage."
    },
    "CLTV": {
        "status": "REMOVE: data leakage",
        "category": "Customer Value",
        "reason": "Customer Lifetime Value in this dataset is calculated retrospectively across the full customer lifecycle, introducing lookahead bias. Conservative approach excludes it."
    },
    "Churn Reason": {
        "status": "REMOVE: data leakage",
        "category": "Churn Information",
        "reason": "Post-churn exit survey reason collected only after a customer has cancelled; unavailable for active customers at prediction time."
    },
    
    # Approved model features (19 features)
    "Gender": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Demographic",
        "reason": "Customer demographic attribute."
    },
    "Senior Citizen": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Demographic",
        "reason": "Senior citizen status (Yes/No)."
    },
    "Partner": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Demographic",
        "reason": "Partner/marital status (Yes/No)."
    },
    "Dependents": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Demographic",
        "reason": "Dependent family status (Yes/No)."
    },
    "Tenure Months": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Account / Tenure",
        "reason": "Number of months subscribed; key indicator of customer lifecycle stage and loyalty."
    },
    "Phone Service": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Subscription to landline phone service (Yes/No)."
    },
    "Multiple Lines": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Multiple phone lines subscription status."
    },
    "Internet Service": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Broadband technology type (DSL, Fiber optic, No)."
    },
    "Online Security": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Add-on cybersecurity subscription status."
    },
    "Online Backup": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Add-on cloud backup subscription status."
    },
    "Device Protection": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Add-on equipment protection plan status."
    },
    "Tech Support": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Add-on technical support subscription status."
    },
    "Streaming TV": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Television streaming service usage."
    },
    "Streaming Movies": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Service Information",
        "reason": "Movie streaming service usage."
    },
    "Contract": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Account / Billing",
        "reason": "Contract agreement type (Month-to-month, One year, Two year); primary indicator of customer commitment."
    },
    "Paperless Billing": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Account / Billing",
        "reason": "Paperless billing enrollment (Yes/No)."
    },
    "Payment Method": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Account / Billing",
        "reason": "Payment method channel used by customer."
    },
    "Monthly Charges": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Account / Billing",
        "reason": "Current monthly billing amount ($)."
    },
    "Total Charges": {
        "status": "KEEP AS MODEL FEATURE",
        "category": "Account / Billing",
        "reason": "Cumulative historical charges billed ($)."
    }
}


def load_clean_data(file_path: str | Path = "data/processed/telco_churn_clean.csv") -> pd.DataFrame:
    """Loads the cleaned Telco dataset from CSV."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {path.resolve()}")
    return pd.read_csv(path)


def select_features(
    df: pd.DataFrame,
    target_col: str = "Churn Value",
    id_col: str = "CustomerID"
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Extracts approved model features (X), target (y), and customer IDs (customer_ids).
    Returns X, y, customer_ids, and summary metadata.
    """
    approved_features = [
        col for col, meta in COLUMN_CLASSIFICATION.items()
        if meta["status"] == "KEEP AS MODEL FEATURE" and col in df.columns
    ]
    
    removed_columns = {
        col: meta["status"] for col, meta in COLUMN_CLASSIFICATION.items()
        if meta["status"].startswith("REMOVE") and col in df.columns
    }
    
    X = df[approved_features].copy()
    y = df[target_col].copy()
    customer_ids = df[id_col].copy() if id_col in df.columns else pd.Series(dtype=str)
    
    metadata = {
        "total_columns": len(df.columns),
        "target_col": target_col,
        "selected_feature_count": len(approved_features),
        "selected_features": approved_features,
        "removed_columns": removed_columns,
    }
    
    return X, y, customer_ids, metadata


def display_feature_selection_summary():
    """Executes feature selection and prints a detailed decision report."""
    df = load_clean_data()
    X, y, customer_ids, meta = select_features(df)
    
    print("=" * 80)
    print("FEATURE SELECTION & DATA LEAKAGE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Original Number of Columns: {meta['total_columns']}")
    print(f"Target Column Name:         {meta['target_col']}")
    print(f"Selected Feature Count:     {meta['selected_feature_count']}")
    print("-" * 80)
    print("SELECTED MODEL FEATURES (19):")
    for idx, feat in enumerate(meta['selected_features'], 1):
        cat = COLUMN_CLASSIFICATION[feat]["category"]
        print(f"  {idx:2d}. {feat:<20} [{cat}]")
    print("-" * 80)
    print(f"REMOVED COLUMNS ({len(meta['removed_columns'])}):")
    for col, status in meta['removed_columns'].items():
        reason = COLUMN_CLASSIFICATION[col]["reason"]
        print(f"  - {col:<15} -> {status}")
        print(f"    Reason: {reason}")
    print("=" * 80)
    print(f"Feature Matrix X Shape:     {X.shape}")
    print(f"Target Vector y Shape:      {y.shape}")
    print(f"Customer IDs Shape:         {customer_ids.shape}")
    print("=" * 80)


if __name__ == "__main__":
    display_feature_selection_summary()
