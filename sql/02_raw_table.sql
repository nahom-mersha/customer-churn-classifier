-- The raw layer preserves the original source values.

USE WAREHOUSE CHURN_WH;
USE DATABASE CHURN_DB;
USE SCHEMA RAW;

CREATE TABLE IF NOT EXISTS TELCO_CHURN_RAW (
    "customerID" VARCHAR,
    "gender" VARCHAR,
    "SeniorCitizen" NUMBER,
    "Partner" VARCHAR,
    "Dependents" VARCHAR,
    "tenure" NUMBER,
    "PhoneService" VARCHAR,
    "MultipleLines" VARCHAR,
    "InternetService" VARCHAR,
    "OnlineSecurity" VARCHAR,
    "OnlineBackup" VARCHAR,
    "DeviceProtection" VARCHAR,
    "TechSupport" VARCHAR,
    "StreamingTV" VARCHAR,
    "StreamingMovies" VARCHAR,
    "Contract" VARCHAR,
    "PaperlessBilling" VARCHAR,
    "PaymentMethod" VARCHAR,
    "MonthlyCharges" NUMBER(10, 2),
    "TotalCharges" VARCHAR,
    "Churn" VARCHAR
);

-- After uploading the CSV to TELCO_STAGE, run:
--
-- COPY INTO TELCO_CHURN_RAW
-- FROM @TELCO_STAGE
-- FILE_FORMAT = (FORMAT_NAME = TELCO_CSV_FORMAT)
-- ON_ERROR = 'ABORT_STATEMENT';

SELECT COUNT(*) AS raw_rows
FROM TELCO_CHURN_RAW;