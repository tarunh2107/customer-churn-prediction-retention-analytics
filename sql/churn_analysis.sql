-- ============================================================================
-- SQL BUSINESS ANALYSIS QUERIES: CUSTOMER CHURN & RETENTION ANALYTICS
-- Database: data/processed/customer_churn.db
-- Table: customer_churn (N = 7,043)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY A: Overall Customer Portfolio KPIs
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charges,
    ROUND(AVG("Tenure Months"), 2) AS avg_tenure_months
FROM customer_churn;


-- ----------------------------------------------------------------------------
-- QUERY B: Customer Churn by Contract Agreement Type
-- ----------------------------------------------------------------------------
SELECT
    "Contract" AS contract_type,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct
FROM customer_churn
GROUP BY "Contract"
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY C: Customer Churn by Internet Service Tier
-- ----------------------------------------------------------------------------
SELECT
    "Internet Service" AS internet_service_tier,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct
FROM customer_churn
GROUP BY "Internet Service"
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY D: Customer Churn by Billing Payment Method
-- ----------------------------------------------------------------------------
SELECT
    "Payment Method" AS payment_method,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN "Churn Value" = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS churn_rate_pct
FROM customer_churn
GROUP BY "Payment Method"
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY E: Customer Churn by Tenure Cohort
-- ----------------------------------------------------------------------------
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


-- ----------------------------------------------------------------------------
-- QUERY F: Analytical Risk Segment Validation & Performance
-- ----------------------------------------------------------------------------
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


-- ----------------------------------------------------------------------------
-- QUERY G: High-Risk Cohort Profile & Driver Prevalence (N = 2,358)
-- ----------------------------------------------------------------------------
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


-- ----------------------------------------------------------------------------
-- QUERY H: Top 20 High-Priority Accounts for Proactive Retention Outreach
-- ----------------------------------------------------------------------------
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


-- ----------------------------------------------------------------------------
-- QUERY I: High-Risk Customers by Contract Type
-- ----------------------------------------------------------------------------
SELECT
    "Contract" AS contract_type,
    COUNT(*) AS high_risk_customer_count,
    ROUND(AVG("Churn Probability") * 100, 2) AS avg_churn_probability_pct,
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS actual_churn_rate_pct
FROM customer_churn
WHERE "Risk Level" = 'High Risk'
GROUP BY "Contract"
ORDER BY high_risk_customer_count DESC;


-- ----------------------------------------------------------------------------
-- QUERY J: High-Risk Customers by Payment Method
-- ----------------------------------------------------------------------------
SELECT
    "Payment Method" AS payment_method,
    COUNT(*) AS high_risk_customer_count,
    ROUND(AVG("Churn Probability") * 100, 2) AS avg_churn_probability_pct,
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS actual_churn_rate_pct
FROM customer_churn
WHERE "Risk Level" = 'High Risk'
GROUP BY "Payment Method"
ORDER BY high_risk_customer_count DESC;
