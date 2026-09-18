from customer_churn_classifier.snowflake_data import load_curated_data

dataframe = load_curated_data()

print("Curated view loaded successfully.")
print(f"Rows loaded: {len(dataframe)}")
print(f"Columns: {len(dataframe.columns)}")
print(dataframe.head())
