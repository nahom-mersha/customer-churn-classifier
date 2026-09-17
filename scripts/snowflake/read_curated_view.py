import os

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

connection = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    role=os.environ["SNOWFLAKE_ROLE"],
)

try:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT *
            FROM ANALYTICS.TELCO_CHURN_CURATED
            """
        )

        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]

        dataframe = pd.DataFrame(rows, columns=columns)

        print("Curated view loaded successfully.")
        print(f"Rows loaded: {len(dataframe)}")
        print(f"Columns: {len(dataframe.columns)}")
        print(dataframe.head())

finally:
    connection.close()
