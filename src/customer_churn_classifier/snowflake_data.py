import os

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()


def get_snowflake_connection():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
    )


def load_curated_data(limit: int | None = None) -> pd.DataFrame:
    query = """
        SELECT *
        FROM ANALYTICS.TELCO_CHURN_CURATED
    """

    if limit is not None:
        query += f"\nLIMIT {int(limit)}"

    connection = get_snowflake_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]

            return pd.DataFrame(rows, columns=columns)
    finally:
        connection.close()
