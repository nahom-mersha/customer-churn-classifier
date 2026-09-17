-- Curated layer:
-- reproduce the existing Python cleaning contract and add
-- analysis-friendly fields.

USE WAREHOUSE CHURN_WH;
USE DATABASE CHURN_DB;

CREATE OR REPLACE VIEW ANALYTICS.TELCO_CHURN_CURATED AS
WITH typed AS (
    SELECT
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",

        TRY_TO_DECIMAL(
            NULLIF(TRIM("TotalCharges"), ''),
            12,
            2
        )::FLOAT AS "TotalCharges",

        CASE "Churn"
            WHEN 'Yes' THEN 1
            WHEN 'No' THEN 0
            ELSE NULL
        END AS "Churn"

    FROM RAW.TELCO_CHURN_RAW
)

SELECT
    typed.*,

    CASE
        WHEN "tenure" < 12 THEN '0-11 months'
        WHEN "tenure" < 24 THEN '12-23 months'
        WHEN "tenure" < 48 THEN '24-47 months'
        ELSE '48+ months'
    END AS tenure_band,

    CASE
        WHEN "MonthlyCharges" < 40 THEN 'Low'
        WHEN "MonthlyCharges" < 80 THEN 'Medium'
        ELSE 'High'
    END AS monthly_charge_band

FROM typed;

-- Basic quality checks
SELECT COUNT(*) AS curated_rows
FROM ANALYTICS.TELCO_CHURN_CURATED;

SELECT
    "Churn",
    COUNT(*) AS customers
FROM ANALYTICS.TELCO_CHURN_CURATED
GROUP BY "Churn"
ORDER BY "Churn";