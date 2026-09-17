import os

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
            SELECT
                CURRENT_USER(),
                CURRENT_ROLE(),
                CURRENT_DATABASE(),
                CURRENT_SCHEMA(),
                CURRENT_WAREHOUSE()
            """
        )
        result = cursor.fetchone()
        if result is None:
            raise RuntimeError("The Snowflake query returned no result.")
        print("Snowflake connection succeeded.")
        print(f"User: {result[0]}")
        print(f"Role: {result[1]}")
        print(f"Database: {result[2]}")
        print(f"Schema: {result[3]}")
        print(f"Warehouse: {result[4]}")
finally:
    connection.close()
