"""
Customer Churn Probability Scoring & Risk Segmentation Module.

This module scores the complete Telco customer base using the preferred Logistic Regression pipeline,
assigns analytical risk tiers (Low, Medium, High Risk), produces actionable prioritization datasets,
and validates monotonic risk alignment with empirical churn rates.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Import feature selection module
try:
    from src.feature_selection import select_features, load_clean_data
except ImportError:
    from feature_selection import select_features, load_clean_data


def assign_risk_tier(probability: float) -> str:
    """
    Categorizes a churn probability into an analytical risk tier.
    - Low Risk:    p < 0.30
    - Medium Risk: 0.30 <= p < 0.60
    - High Risk:   p >= 0.60
    """
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


def score_and_segment_customers(
    data_path: str | Path = "data/processed/telco_churn_clean.csv",
    model_path: str | Path = "models/logistic_regression_pipeline.joblib",
    output_path: str | Path = "outputs/predictions/customer_churn_risk_predictions.csv",
    threshold_high_risk: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads full dataset, generates churn probabilities, adds analytical risk segments,
    exports sorted prediction dataset, and computes validation metrics.
    """
    print("Loading cleaned dataset and trained Logistic Regression pipeline...")
    df = load_clean_data(data_path)
    model = joblib.load(model_path)
    
    # Extract feature matrix X
    X, y, customer_ids, meta = select_features(df)
    
    # Generate probabilities
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add scoring columns
    df["Churn Probability"] = probabilities.round(4)
    df["Risk Level"] = df["Churn Probability"].apply(assign_risk_tier)
    df["Predicted Churn"] = (df["Churn Probability"] >= threshold_high_risk).astype(int)
    
    # Select business & dashboard columns
    output_columns = [
        "CustomerID",
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Internet Service",
        "Contract",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
        "Tech Support",
        "Online Security",
        "Paperless Billing",
        "Churn Value",
        "Churn Label",
        "Churn Probability",
        "Risk Level",
        "Predicted Churn",
    ]
    
    # Filter and sort descending by probability
    scored_df = df[output_columns].sort_values(by="Churn Probability", ascending=False).reset_index(drop=True)
    
    # Export CSV
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out_file, index=False)
    print(f"Successfully saved scored customer predictions to: {out_file.resolve()}")
    
    # Aggregate validation statistics by Risk Level
    validation_summary = scored_df.groupby("Risk Level").agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Churn_Probability=("Churn Probability", lambda x: round(x.mean() * 100, 2)),
        Actual_Churned_Count=("Churn Value", "sum"),
        Actual_Churn_Rate_Pct=("Churn Value", lambda x: round(x.mean() * 100, 2)),
        Avg_Monthly_Charges=("Monthly Charges", lambda x: round(x.mean(), 2)),
        Avg_Tenure_Months=("Tenure Months", lambda x: round(x.mean(), 1))
    ).reindex(["Low Risk", "Medium Risk", "High Risk"])
    
    validation_summary["Percentage_of_Total_Base"] = (
        validation_summary["Customer_Count"] / len(scored_df) * 100
    ).round(2)
    
    return scored_df, validation_summary


def display_segmentation_summary():
    """Runs segmentation pipeline and prints diagnostic breakdown."""
    scored_df, val_summary = score_and_segment_customers()
    
    print("\n" + "=" * 95)
    print("CUSTOMER CHURN RISK SEGMENTATION SUMMARY (FULL DATASET, N = 7,043)")
    print("=" * 95)
    print(val_summary.to_string())
    print("=" * 95)
    
    # Top 10 High Risk Customers
    print("\nTOP 10 HIGHEST-RISK CUSTOMERS:")
    top_10 = scored_df.head(10)[["CustomerID", "Churn Probability", "Risk Level", "Contract", "Tenure Months", "Monthly Charges", "Churn Value"]]
    print(top_10.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    display_segmentation_summary()
