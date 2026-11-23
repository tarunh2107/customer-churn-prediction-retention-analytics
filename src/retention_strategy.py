"""
Rule-Based Customer Retention Strategy & Prioritization Module.

This module consumes the scored customer churn predictions, identifies transparent,
observable risk drivers for each customer, applies heuristic retention action rules,
calculates customer priority rankings, and exports a business-ready action dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def compute_monthly_charges_threshold(df: pd.DataFrame, quantile: float = 0.75) -> float:
    """
    Computes a data-driven threshold for high monthly charges using the 75th percentile.
    """
    threshold = float(df["Monthly Charges"].quantile(quantile))
    return round(threshold, 2)


def identify_risk_drivers(row: pd.Series, high_charges_threshold: float) -> str:
    """
    Identifies observable, non-leaking customer characteristics that explain churn risk.
    """
    drivers = []
    
    # 1. Contract commitment
    if row["Contract"] == "Month-to-month":
        drivers.append("Month-to-month contract")
        
    # 2. Tenure maturity
    if row["Tenure Months"] <= 12:
        drivers.append("Short tenure (<= 12m)")
        
    # 3. Monthly billing magnitude
    if row["Monthly Charges"] >= high_charges_threshold:
        drivers.append(f"High monthly charges (>= ${high_charges_threshold:.2f})")
        
    # 4. Service support add-on
    if row["Tech Support"] == "No":
        drivers.append("No tech support")
        
    # 5. Online security add-on
    if row["Online Security"] == "No":
        drivers.append("No online security")
        
    # 6. Payment channel
    if row["Payment Method"] == "Electronic check":
        drivers.append("Electronic check payment")
        
    # 7. Internet technology
    if row["Internet Service"] == "Fiber optic":
        drivers.append("Fiber optic service")
        
    return "; ".join(drivers) if drivers else "Standard account profile"


def determine_retention_actions(row: pd.Series, high_charges_threshold: float) -> str:
    """
    Assigns rule-based, heuristic retention recommendations.
    Multiple matching actions are combined for comprehensive account triage.
    """
    risk_level = row["Risk Level"]
    
    if risk_level == "High Risk":
        actions = []
        
        # Rule 1: Month-to-month contract -> incentivize longer-term commitment
        if row["Contract"] == "Month-to-month":
            actions.append("Contract Migration (Offer Longer-Term Contract Incentive/Options)")
            
        # Rule 2: Early lifecycle account -> proactive onboarding
        if row["Tenure Months"] <= 12:
            actions.append("Early-Lifecycle Customer Success Check-In")
            
        # Rule 3: High monthly charges -> plan review
        if row["Monthly Charges"] >= high_charges_threshold:
            actions.append("Plan Optimization & Pricing Tier Review")
            
        # Rule 4: Missing Tech Support -> technical assistance outreach
        if row["Tech Support"] == "No":
            actions.append("Proactive Tech Support Onboarding & Assistance")
            
        # Rule 5: Missing Online Security -> value-add bundle
        if row["Online Security"] == "No":
            actions.append("Security & Value-Add Service Bundle Engagement")
            
        # Rule 6: Electronic check -> auto-pay incentive
        if row["Payment Method"] == "Electronic check":
            actions.append("Encourage Automated Payment / Auto-Pay Setup")
            
        return "; ".join(actions) if actions else "Priority Account Management Check-In"
        
    elif risk_level == "Medium Risk":
        return "Targeted Engagement & Account Health Check-In (Address Specific Risk Drivers)"
        
    else:  # Low Risk
        return "Standard Service Maintenance & Loyalty Program Engagement"


def build_retention_strategy_dataset(
    input_path: str | Path = "outputs/predictions/customer_churn_risk_predictions.csv",
    output_path: str | Path = "outputs/predictions/customer_retention_recommendations.csv"
) -> tuple[pd.DataFrame, dict]:
    """
    Loads scored predictions, appends transparent risk drivers and composite retention actions,
    calculates priority ranks, and exports the final dataset.
    """
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input predictions file not found at: {in_file.resolve()}")
        
    print(f"Loading scored predictions from: {in_file.resolve()}")
    df = pd.read_csv(in_file)
    
    # 1. Compute empirical 75th percentile for Monthly Charges
    high_charge_threshold = compute_monthly_charges_threshold(df, quantile=0.75)
    print(f"Calculated 75th Percentile Monthly Charges Threshold: ${high_charge_threshold:.2f}")
    
    # 2. Generate Risk Drivers and Retention Actions
    df["Risk Drivers"] = df.apply(lambda r: identify_risk_drivers(r, high_charge_threshold), axis=1)
    df["Retention Action"] = df.apply(lambda r: determine_retention_actions(r, high_charge_threshold), axis=1)
    
    # 3. Create Priority Rank (High Risk first, then Medium, then Low; sorted by probability descending)
    risk_rank_map = {"High Risk": 1, "Medium Risk": 2, "Low Risk": 3}
    df["_tier_rank"] = df["Risk Level"].map(risk_rank_map)
    df = df.sort_values(
        by=["_tier_rank", "Churn Probability", "Monthly Charges"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    
    df["Priority Rank"] = range(1, len(df) + 1)
    df.drop(columns=["_tier_rank"], inplace=True)
    
    # Rearrange column order with Priority Rank first
    cols = ["Priority Rank", "CustomerID", "Churn Probability", "Risk Level", "Predicted Churn", "Risk Drivers", "Retention Action"] + [
        c for c in df.columns if c not in ["Priority Rank", "CustomerID", "Churn Probability", "Risk Level", "Predicted Churn", "Risk Drivers", "Retention Action"]
    ]
    df = df[cols]
    
    # Export CSV
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Successfully exported final retention recommendations to: {out_file.resolve()}")
    
    # Diagnostics & Aggregation
    high_risk_df = df[df["Risk Level"] == "High Risk"]
    diagnostics = {
        "total_customers": len(df),
        "total_high_risk": len(high_risk_df),
        "high_risk_pct": round(len(high_risk_df) / len(df) * 100, 2),
        "high_risk_m2m": int((high_risk_df["Contract"] == "Month-to-month").sum()),
        "high_risk_no_tech_support": int((high_risk_df["Tech Support"] == "No").sum()),
        "high_risk_no_online_sec": int((high_risk_df["Online Security"] == "No").sum()),
        "high_risk_short_tenure": int((high_risk_df["Tenure Months"] <= 12).sum()),
        "high_risk_high_charges": int((high_risk_df["Monthly Charges"] >= high_charge_threshold).sum()),
        "high_risk_elec_check": int((high_risk_df["Payment Method"] == "Electronic check").sum()),
        "high_risk_fiber": int((high_risk_df["Internet Service"] == "Fiber optic").sum()),
        "high_charge_threshold": high_charge_threshold,
    }
    
    return df, diagnostics


def display_retention_summary():
    """Executes retention pipeline and prints summary statistics."""
    df, stats = build_retention_strategy_dataset()
    
    print("\n" + "=" * 95)
    print("CUSTOMER RETENTION STRATEGY & RISK DRIVER SUMMARY")
    print("=" * 95)
    print(f"Total Customer Accounts:                        {stats['total_customers']:,}")
    print(f"High-Risk Prioritization Segment:               {stats['total_high_risk']:,} ({stats['high_risk_pct']}%)")
    print(f"Data-Driven High Monthly Charges Cutoff (Q3):   ${stats['high_charge_threshold']:.2f}")
    print("-" * 95)
    print("HIGH-RISK COHORT DRIVER PREVALENCE (N = 2,358):")
    print(f"  - Month-to-Month Contracts:                   {stats['high_risk_m2m']:,} ({(stats['high_risk_m2m']/stats['total_high_risk'])*100:.2f}%)")
    print(f"  - Lacking Online Security Add-On:             {stats['high_risk_no_online_sec']:,} ({(stats['high_risk_no_online_sec']/stats['total_high_risk'])*100:.2f}%)")
    print(f"  - Lacking Tech Support Add-On:                {stats['high_risk_no_tech_support']:,} ({(stats['high_risk_no_tech_support']/stats['total_high_risk'])*100:.2f}%)")
    print(f"  - Fiber Optic Internet Subscribers:           {stats['high_risk_fiber']:,} ({(stats['high_risk_fiber']/stats['total_high_risk'])*100:.2f}%)")
    print(f"  - Electronic Check Payment Channel:           {stats['high_risk_elec_check']:,} ({(stats['high_risk_elec_check']/stats['total_high_risk'])*100:.2f}%)")
    print(f"  - Short Tenure (<= 12 Months):                {stats['high_risk_short_tenure']:,} ({(stats['high_risk_short_tenure']/stats['total_high_risk'])*100:.2f}%)")
    print(f"  - High Monthly Charges (>= ${stats['high_charge_threshold']:.2f}):          {stats['high_risk_high_charges']:,} ({(stats['high_risk_high_charges']/stats['total_high_risk'])*100:.2f}%)")
    print("=" * 95)
    
    print("\nTOP 10 PRIORITY CUSTOMERS FOR RETENTION OUTREACH:")
    top_10 = df.head(10)[["Priority Rank", "CustomerID", "Churn Probability", "Risk Level", "Risk Drivers", "Retention Action"]]
    for _, row in top_10.iterrows():
        print(f"\n[Rank {row['Priority Rank']:2d}] Customer ID: {row['CustomerID']} | Churn Probability: {row['Churn Probability']:.4f} | Tier: {row['Risk Level']}")
        print(f"  Drivers: {row['Risk Drivers']}")
        print(f"  Action:  {row['Retention Action']}")
    print("=" * 95)


if __name__ == "__main__":
    display_retention_summary()
