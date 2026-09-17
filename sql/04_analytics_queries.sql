USE WAREHOUSE CHURN_WH;
USE DATABASE CHURN_DB;

-- Business question:
-- Which contract types have the highest observed churn?

SELECT
    "Contract" AS contract_type,
    COUNT(*) AS customers,
    SUM("Churn") AS churned_customers,
    AVG("Churn") AS churn_rate,
    AVG("MonthlyCharges") AS average_monthly_charges
FROM ANALYTICS.TELCO_CHURN_CURATED
GROUP BY "Contract"
ORDER BY churn_rate DESC;


-- Business question:
-- How does churn change across tenure bands?

SELECT
    tenure_band,
    COUNT(*) AS customers,
    AVG("Churn") AS churn_rate
FROM ANALYTICS.TELCO_CHURN_CURATED
GROUP BY tenure_band
ORDER BY tenure_band;


-- Window function:
-- Rank payment methods within each contract by churn rate.

WITH payment_summary AS (
    SELECT
        "Contract" AS contract_type,
        "PaymentMethod" AS payment_method,
        COUNT(*) AS customers,
        AVG("Churn") AS churn_rate
    FROM ANALYTICS.TELCO_CHURN_CURATED
    GROUP BY "Contract", "PaymentMethod"
)

SELECT
    payment_summary.*,
    DENSE_RANK() OVER (
        PARTITION BY contract_type
        ORDER BY churn_rate DESC
    ) AS churn_rate_rank
FROM payment_summary
ORDER BY contract_type, churn_rate_rank;


-- High-risk customer segments.
-- Only include segments with at least 25 customers.

SELECT
    "Contract" AS contract_type,
    tenure_band,
    "InternetService" AS internet_service,
    COUNT(*) AS customers,
    AVG("Churn") AS churn_rate,
    SUM("MonthlyCharges") AS monthly_revenue_at_risk
FROM ANALYTICS.TELCO_CHURN_CURATED
GROUP BY
    "Contract",
    tenure_band,
    "InternetService"
HAVING COUNT(*) >= 25
ORDER BY
    churn_rate DESC,
    monthly_revenue_at_risk DESC;