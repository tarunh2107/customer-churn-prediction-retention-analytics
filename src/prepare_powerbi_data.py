"""
Power BI Dashboard Dataset Preparation Module.

This module processes the final retention recommendations dataset, generates
Power BI-optimized analytical dimensions (Tenure Groups, Charge Bands, Status Labels, Risk Sorting Order),
validates data integrity, and exports clean dashboard data files.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def get_tenure_group(months: int) -> str:
    """Classifies tenure in months into standardized cohorts."""
    if months <= 12:
        return "0-12 Months"
    elif months <= 24:
        return "13-24 Months"
    elif months <= 36:
        return "25-36 Months"
    elif months <= 48:
        return "37-48 Months"
    elif months <= 60:
        return "49-60 Months"
    else:
        return "61+ Months"


def get_monthly_charge_band(charge: float, q25: float, q50: float, q75: float) -> str:
    """Classifies monthly billing amount into data-driven quartile bands."""
    if charge <= q25:
        return f"Low Charges (<=${q25:.2f})"
    elif charge <= q50:
        return f"Lower-Mid Charges (${q25:.2f}-${q50:.2f})"
    elif charge <= q75:
        return f"Upper-Mid Charges (${q50:.2f}-${q75:.2f})"
    else:
        return f"High Charges (>${q75:.2f})"


def prepare_powerbi_datasets(
    input_path: str | Path = "outputs/predictions/customer_retention_recommendations.csv",
    output_dir: str | Path = "powerbi"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Creates Power BI dashboard datasets and performs data integrity validations.
    """
    in_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not in_file.exists():
        raise FileNotFoundError(f"Input predictions file not found at: {in_file.resolve()}")
        
    print(f"Loading retention recommendations from: {in_file.resolve()}...")
    df = pd.read_csv(in_file)
    
    # 1. Compute Quartiles for Monthly Charges
    q25 = float(df["Monthly Charges"].quantile(0.25))
    q50 = float(df["Monthly Charges"].quantile(0.50))
    q75 = float(df["Monthly Charges"].quantile(0.75))
    min_charge = float(df["Monthly Charges"].min())
    max_charge = float(df["Monthly Charges"].max())
    
    print(f"Monthly Charges Quartiles: Min=${min_charge:.2f}, Q1=${q25:.2f}, Median=${q50:.2f}, Q3=${q75:.2f}, Max=${max_charge:.2f}")
    
    # 2. Add Analytical Dimension Columns
    df["Tenure Group"] = df["Tenure Months"].apply(get_tenure_group)
    df["Monthly Charge Band"] = df["Monthly Charges"].apply(
        lambda c: get_monthly_charge_band(c, q25, q50, q75)
    )
    df["Churn Status"] = df["Churn Value"].apply(lambda v: "Churned" if v == 1 else "Retained")
    df["Predicted Churn Status"] = df["Predicted Churn"].apply(
        lambda v: "Predicted Churn" if v == 1 else "Predicted Retained"
    )
    
    risk_sort_map = {"Low Risk": 1, "Medium Risk": 2, "High Risk": 3}
    df["Risk Sort Order"] = df["Risk Level"].map(risk_sort_map)
    
    # 3. Integrity Validations
    assert len(df) == 7043, f"Row count error: {len(df)} != 7043"
    assert df["CustomerID"].nunique() == 7043, f"Unique customer count error: {df['CustomerID'].nunique()} != 7043"
    assert df.duplicated(subset=["CustomerID"]).sum() == 0, "Duplicate customers found!"
    assert df["Churn Probability"].isna().sum() == 0, "Missing churn probabilities found!"
    assert df["Risk Level"].isna().sum() == 0, "Missing risk levels found!"
    assert df["Priority Rank"].isna().sum() == 0, "Missing priority ranks found!"
    print("All 6 data integrity validations passed successfully.")
    
    # 4. Save Main Dashboard Dataset
    main_output_path = out_dir / "customer_churn_dashboard_data.csv"
    df.to_csv(main_output_path, index=False)
    print(f"Exported main dashboard dataset to: {main_output_path.resolve()}")
    
    # 5. Create KPI Summary Dataset
    kpi_df = pd.DataFrame([{
        "Total Customers": len(df),
        "Total Churned": int((df["Churn Value"] == 1).sum()),
        "Overall Churn Rate": round((df["Churn Value"] == 1).mean() * 100, 2),
        "High Risk Customers": int((df["Risk Level"] == "High Risk").sum()),
        "Medium Risk Customers": int((df["Risk Level"] == "Medium Risk").sum()),
        "Low Risk Customers": int((df["Risk Level"] == "Low Risk").sum()),
        "Average Monthly Charges": round(df["Monthly Charges"].mean(), 2),
        "Average Tenure": round(df["Tenure Months"].mean(), 2),
        "Average Churn Probability": round(df["Churn Probability"].mean() * 100, 2)
    }])
    kpi_output_path = out_dir / "customer_churn_kpi_summary.csv"
    kpi_df.to_csv(kpi_output_path, index=False)
    print(f"Exported KPI summary dataset to: {kpi_output_path.resolve()}")
    
    # 6. Create High-Risk Action Table
    high_risk_cols = [
        "Priority Rank",
        "CustomerID",
        "Churn Probability",
        "Contract",
        "Tenure Months",
        "Internet Service",
        "Payment Method",
        "Monthly Charges",
        "Tech Support",
        "Online Security",
        "Risk Drivers",
        "Retention Action"
    ]
    high_risk_df = df[df["Risk Level"] == "High Risk"][high_risk_cols].sort_values(
        by="Priority Rank", ascending=True
    ).reset_index(drop=True)
    
    high_risk_output_path = out_dir / "high_risk_customers.csv"
    high_risk_df.to_csv(high_risk_output_path, index=False)
    print(f"Exported High-Risk action table to: {high_risk_output_path.resolve()}")
    
    metadata = {
        "total_rows": len(df),
        "unique_customers": df["CustomerID"].nunique(),
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "low_risk_count": int((df["Risk Level"] == "Low Risk").sum()),
        "medium_risk_count": int((df["Risk Level"] == "Medium Risk").sum()),
        "high_risk_count": int((df["Risk Level"] == "High Risk").sum()),
        "kpi_summary": kpi_df.to_dict(orient="records")[0]
    }
    
    return df, kpi_df, high_risk_df, metadata


def display_powerbi_summary():
    """Runs data preparation and displays summary metrics."""
    df, kpi_df, high_risk_df, meta = prepare_powerbi_datasets()
    
    print("\n" + "=" * 90)
    print("POWER BI DATASET PREPARATION SUMMARY")
    print("=" * 90)
    print(f"Total Rows in Main Dataset:        {meta['total_rows']:,}")
    print(f"Unique Customer IDs:               {meta['unique_customers']:,}")
    print(f"High-Risk Segment Size:            {meta['high_risk_count']:,} ({meta['high_risk_count']/meta['total_rows']*100:.2f}%)")
    print(f"Medium-Risk Segment Size:          {meta['medium_risk_count']:,} ({meta['medium_risk_count']/meta['total_rows']*100:.2f}%)")
    print(f"Low-Risk Segment Size:             {meta['low_risk_count']:,} ({meta['low_risk_count']/meta['total_rows']*100:.2f}%)")
    print("-" * 90)
    print("DATA-DRIVEN MONTHLY CHARGE QUARTILES:")
    print(f"  - Q1 (25th Percentile):           ${meta['q25']:.2f}")
    print(f"  - Q2 (50th Percentile / Median):  ${meta['q50']:.2f}")
    print(f"  - Q3 (75th Percentile):           ${meta['q75']:.2f}")
    print("-" * 90)
    print("PORTFOLIO KPI SUMMARY:")
    for k, v in meta["kpi_summary"].items():
        unit = "%" if "Rate" in k or "Probability" in k else ("$" if "Charges" in k else "")
        print(f"  - {k:<28}: {unit}{v}")
    print("=" * 90)


if __name__ == "__main__":
    display_powerbi_summary()
