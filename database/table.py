import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "password": os.environ["DB_PASSWORD"],
    "port": os.environ["DB_PORT"],
    "user": os.environ["DB_USER"],
    "database": os.environ["DB_NAME"],
    "sslmode": "require",
}

print("Connecting to:", DB_CONFIG["host"], "| database:", DB_CONFIG["database"])

conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = False

try:
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_database();")
        print("Actually connected to DB:", cursor.fetchone())

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.financial_metrics (
                id SERIAL PRIMARY KEY,
                company VARCHAR(100),
                year VARCHAR(10),
                revenue TEXT,
                net_income TEXT,
                operating_income TEXT,
                cash_flow TEXT,
                total_assets TEXT,
                total_liabilities TEXT,
                risk_factors TEXT,
                growth_drivers TEXT
            );
        """)
    conn.commit()
    print("CREATE TABLE executed and committed.")

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_name = 'financial_metrics';
        """)
        print("Verification inside Python:", cursor.fetchall())

except Exception as e:
    conn.rollback()
    print("ERROR:", e)
finally:
    conn.close()