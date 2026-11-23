"""
SQLite Database Ingestion & SQL Business Analytics Runner.

This module initializes the SQLite database at data/processed/customer_churn.db,
loads the prioritized customer retention dataset into the 'customer_churn' table,
executes queries defined in sql/churn_analysis.sql, exports analytical summary CSVs,
and verifies mathematical alignment with earlier Python EDA.
"""

from pathlib import Path
import sqlite3
import pandas as pd


def initialize_sqlite_db(
    csv_path: str | Path = "outputs/predictions/customer_retention_recommendations.csv",
    db_path: str | Path = "data/processed/customer_churn.db",
    table_name: str = "customer_churn"
) -> sqlite3.Connection:
    """
    Loads customer retention recommendations CSV and creates/replaces table in SQLite.
    """
    csv_file = Path(csv_path)
    db_file = Path(db_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Input CSV not found at {csv_file.resolve()}")
        
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading {csv_file.resolve()}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns.")
    
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Successfully loaded table '{table_name}' into SQLite database: {db_file.resolve()}")
    
    return conn


def execute_sql_analytics(
    conn: sqlite3.Connection,
    output_dir: str | Path = "outputs"
) -> dict:
    """
    Executes business queries against SQLite database, exports CSV reports, and returns results.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    queries = {
        "overall_kpis": {
            "file": out_dir / "sql_overall_kpis.csv",
            "sql": """
                SELECT
                    COUNT(*) AS total_customers,
                    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_customers,
                    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct,
                    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charges,
                    ROUND(AVG("Tenure Months"), 2) AS avg_tenure_months
                FROM customer_churn;
            """
        },
        "churn_by_contract": {
            "file": out_dir / "sql_churn_by_contract.csv",
            "sql": """
                SELECT
                    "Contract" AS contract_type,
                    COUNT(*) AS customer_count,
                    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_count,
                    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct
                FROM customer_churn
                GROUP BY "Contract"
                ORDER BY churn_rate_pct DESC;
            """
        },
        "churn_by_internet_service": {
            "file": out_dir / "sql_churn_by_internet_service.csv",
            "sql": """
                SELECT
                    "Internet Service" AS internet_service_tier,
                    COUNT(*) AS customer_count,
                    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_count,
                    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct
                FROM customer_churn
                GROUP BY "Internet Service"
                ORDER BY churn_rate_pct DESC;
            """
        },
        "churn_by_payment_method": {
            "file": out_dir / "sql_churn_by_payment_method.csv",
            "sql": """
                SELECT
                    "Payment Method" AS payment_method,
                    COUNT(*) AS customer_count,
                    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_count,
                    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct
                FROM customer_churn
                GROUP BY "Payment Method"
                ORDER BY churn_rate_pct DESC;
            """
        },
        "churn_by_tenure": {
            "file": out_dir / "sql_churn_by_tenure.csv",
            "sql": """
                SELECT
                    CASE
                        WHEN "Tenure Months" <= 12 THEN '0-12 months'
                        WHEN "Tenure Months" <= 24 THEN '13-24 months'
                        WHEN "Tenure Months" <= 36 THEN '25-36 months'
                        WHEN "Tenure Months" <= 48 THEN '37-48 months'
                        WHEN "Tenure Months" <= 60 THEN '49-60 months'
                        ELSE '61+ months'
                    END AS tenure_group,
                    COUNT(*) AS customer_count,
                    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_count,
                    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct
                FROM customer_churn
                GROUP BY tenure_group
                ORDER BY MIN("Tenure Months") ASC;
            """
        },
        "risk_segment_analysis": {
            "file": out_dir / "sql_risk_segment_analysis.csv",
            "sql": """
                SELECT
                    "Risk Level" AS risk_level,
                    COUNT(*) AS customer_count,
                    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS actual_churned_count,
                    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS actual_churn_rate_pct,
                    ROUND(AVG("Churn Probability") * 100, 2) AS avg_churn_probability_pct,
                    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charges,
                    ROUND(AVG("Tenure Months"), 2) AS avg_tenure_months
                FROM customer_churn
                GROUP BY "Risk Level"
                ORDER BY CASE 
                    WHEN "Risk Level" = 'High Risk' THEN 1 
                    WHEN "Risk Level" = 'Medium Risk' THEN 2 
                    ELSE 3 
                END ASC;
            """
        },
        "high_risk_profile": {
            "file": out_dir / "sql_high_risk_profile.csv",
            "sql": """
                SELECT
                    COUNT(*) AS total_high_risk_customers,
                    SUM(CASE WHEN "Contract" = 'Month-to-month' THEN 1 ELSE 0 END) AS month_to_month_count,
                    ROUND(SUM(CASE WHEN "Contract" = 'Month-to-month' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS month_to_month_pct,
                    SUM(CASE WHEN "Tech Support" = 'No' THEN 1 ELSE 0 END) AS no_tech_support_count,
                    ROUND(SUM(CASE WHEN "Tech Support" = 'No' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS no_tech_support_pct,
                    SUM(CASE WHEN "Online Security" = 'No' THEN 1 ELSE 0 END) AS no_online_security_count,
                    ROUND(SUM(CASE WHEN "Online Security" = 'No' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS no_online_security_pct,
                    SUM(CASE WHEN "Payment Method" = 'Electronic check' THEN 1 ELSE 0 END) AS electronic_check_count,
                    ROUND(SUM(CASE WHEN "Payment Method" = 'Electronic check' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS electronic_check_pct,
                    SUM(CASE WHEN "Internet Service" = 'Fiber optic' THEN 1 ELSE 0 END) AS fiber_optic_count,
                    ROUND(SUM(CASE WHEN "Internet Service" = 'Fiber optic' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS fiber_optic_pct,
                    SUM(CASE WHEN "Tenure Months" <= 12 THEN 1 ELSE 0 END) AS short_tenure_count,
                    ROUND(SUM(CASE WHEN "Tenure Months" <= 12 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS short_tenure_pct
                FROM customer_churn
                WHERE "Risk Level" = 'High Risk';
            """
        },
        "top_priority_customers": {
            "file": out_dir / "sql_top_priority_customers.csv",
            "sql": """
                SELECT
                    "Priority Rank" AS priority_rank,
                    "CustomerID" AS customer_id,
                    "Churn Probability" AS churn_probability,
                    "Risk Level" AS risk_level,
                    "Contract" AS contract,
                    "Tenure Months" AS tenure_months,
                    "Monthly Charges" AS monthly_charges,
                    "Risk Drivers" AS risk_drivers,
                    "Retention Action" AS retention_action
                FROM customer_churn
                ORDER BY "Priority Rank" ASC
                LIMIT 20;
            """
        }
    }
    
    results = {}
    for key, item in queries.items():
        res_df = pd.read_sql_query(item["sql"], conn)
        res_df.to_csv(item["file"], index=False)
        results[key] = res_df
        print(f"Exported {item['file'].name} ({len(res_df)} rows).")
        
    return results


def validate_sql_against_python_eda(results: dict):
    """Validates that SQL aggregation results match Python baseline metrics."""
    print("\n" + "=" * 80)
    print("VALIDATION: SQL AGGREGATION VS. PYTHON BASELINE")
    print("=" * 80)
    
    # 1. Overall KPIs
    kpi = results["overall_kpis"].iloc[0]
    total_cust = int(kpi["total_customers"])
    churned_cust = int(kpi["churned_customers"])
    churn_rate = float(kpi["churn_rate_pct"])
    
    assert total_cust == 7043, f"Total customers mismatch: {total_cust} != 7043"
    assert churned_cust == 1869, f"Churned customers mismatch: {churned_cust} != 1869"
    assert abs(churn_rate - 26.54) <= 0.05, f"Churn rate mismatch: {churn_rate} != 26.54%"
    print(f"[PASSED] Overall KPIs: Total={total_cust:,}, Churned={churned_cust:,}, Rate={churn_rate:.2f}%")
    
    # 2. Contract Churn Rates
    contract_df = results["churn_by_contract"].set_index("contract_type")
    m2m_rate = float(contract_df.loc["Month-to-month", "churn_rate_pct"])
    one_yr_rate = float(contract_df.loc["One year", "churn_rate_pct"])
    two_yr_rate = float(contract_df.loc["Two year", "churn_rate_pct"])
    
    assert abs(m2m_rate - 42.71) <= 0.05, f"Month-to-month rate mismatch: {m2m_rate}"
    assert abs(one_yr_rate - 11.27) <= 0.05, f"One year rate mismatch: {one_yr_rate}"
    assert abs(two_yr_rate - 2.83) <= 0.05, f"Two year rate mismatch: {two_yr_rate}"
    print(f"[PASSED] Contract Rates: M2M={m2m_rate:.2f}%, 1-Yr={one_yr_rate:.2f}%, 2-Yr={two_yr_rate:.2f}%")
    
    # 3. Risk Level Actual Churn Rates
    risk_df = results["risk_segment_analysis"].set_index("risk_level")
    low_rate = float(risk_df.loc["Low Risk", "actual_churn_rate_pct"])
    med_rate = float(risk_df.loc["Medium Risk", "actual_churn_rate_pct"])
    high_rate = float(risk_df.loc["High Risk", "actual_churn_rate_pct"])
    
    assert abs(low_rate - 4.91) <= 0.05, f"Low risk rate mismatch: {low_rate}"
    assert abs(med_rate - 22.76) <= 0.05, f"Medium risk rate mismatch: {med_rate}"
    assert abs(high_rate - 58.10) <= 0.05, f"High risk rate mismatch: {high_rate}"
    print(f"[PASSED] Risk Segments: Low={low_rate:.2f}%, Medium={med_rate:.2f}%, High={high_rate:.2f}%")
    print("=" * 80)
    print("ALL SQL BUSINESS AGGREGATIONS VALIDATED SUCCESSFULLY WITH ZERO DISCREPANCIES.")


def run_sql_pipeline():
    """Runs end-to-end database creation, SQL query execution, and validation."""
    conn = initialize_sqlite_db()
    results = execute_sql_analytics(conn)
    validate_sql_against_python_eda(results)
    conn.close()


if __name__ == "__main__":
    run_sql_pipeline()
