# Data Quality Report

## Shape

- Rows: 7043
- Columns: 21

## Target Distribution

`Churn` uses `0 = No churn` and `1 = Churn`.

- `0`: 5174 rows (73.46%)
- `1`: 1869 rows (26.54%)

## Missing Values

- `TotalCharges`: 11

## Duplicate Rows

- Duplicate rows: 0

## Column Data Types

- `customerID`: `str`
- `gender`: `str`
- `SeniorCitizen`: `int64`
- `Partner`: `str`
- `Dependents`: `str`
- `tenure`: `int64`
- `PhoneService`: `str`
- `MultipleLines`: `str`
- `InternetService`: `str`
- `OnlineSecurity`: `str`
- `OnlineBackup`: `str`
- `DeviceProtection`: `str`
- `TechSupport`: `str`
- `StreamingTV`: `str`
- `StreamingMovies`: `str`
- `Contract`: `str`
- `PaperlessBilling`: `str`
- `PaymentMethod`: `str`
- `MonthlyCharges`: `float64`
- `TotalCharges`: `float64`
- `Churn`: `int64`

## Numeric Ranges

- `SeniorCitizen`: min=0, max=1, mean=0.16
- `tenure`: min=0, max=72, mean=32.37
- `MonthlyCharges`: min=18.25, max=118.75, mean=64.76
- `TotalCharges`: min=18.8, max=8684.8, mean=2283.30
- `Churn`: min=0, max=1, mean=0.27

## Categorical Values

- `customerID`: 7590-VHVEG (1), 5575-GNVDE (1), 3668-QPYBK (1), 7795-CFOCW (1), 9237-HQITU (1), 9305-CDSKC (1), 1452-KIOVK (1), 6713-OKOMC (1), 7892-POOKP (1), 6388-TABGU (1)
- `gender`: Male (3555), Female (3488)
- `Partner`: No (3641), Yes (3402)
- `Dependents`: No (4933), Yes (2110)
- `PhoneService`: Yes (6361), No (682)
- `MultipleLines`: No (3390), Yes (2971), No phone service (682)
- `InternetService`: Fiber optic (3096), DSL (2421), No (1526)
- `OnlineSecurity`: No (3498), Yes (2019), No internet service (1526)
- `OnlineBackup`: No (3088), Yes (2429), No internet service (1526)
- `DeviceProtection`: No (3095), Yes (2422), No internet service (1526)
- `TechSupport`: No (3473), Yes (2044), No internet service (1526)
- `StreamingTV`: No (2810), Yes (2707), No internet service (1526)
- `StreamingMovies`: No (2785), Yes (2732), No internet service (1526)
- `Contract`: Month-to-month (3875), Two year (1695), One year (1473)
- `PaperlessBilling`: Yes (4171), No (2872)
- `PaymentMethod`: Electronic check (2365), Mailed check (1612), Bank transfer (automatic) (1544), Credit card (automatic) (1522)

## Validation Checks

- `target_is_binary_0_1`: PASS
- `tenure_non_negative`: PASS
- `monthly_charges_non_negative`: PASS
- `total_charges_non_negative_or_missing`: PASS

## Cleaning Decisions

- `customerID` is kept in the cleaned dataset for reference, but it will not be used as a model feature.
- `TotalCharges` blank strings were converted to missing values and then converted to numeric.
- `Churn` was encoded as `No = 0` and `Yes = 1`.
- Missing `TotalCharges` values are retained for later pipeline imputation instead of dropping rows.