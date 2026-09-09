import os
import json
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

INSERT_SQL = """
    INSERT INTO public.financial_metrics
        (company, year, revenue, net_income, operating_income,
         cash_flow, total_assets, total_liabilities,
         risk_factors, growth_drivers)
    VALUES
        (%(company)s, %(year)s, %(revenue)s, %(net_income)s, %(operating_income)s,
         %(cash_flow)s, %(total_assets)s, %(total_liabilities)s,
         %(risk_factors)s, %(growth_drivers)s)
    RETURNING id;
"""


def _stringify_list(value):
    """Convert a list field (or None) into TEXT for storage."""
    if value is None:
        return None
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)  # keep structure, easy to json.loads back later
    return str(value)


def save_metrics(company: str, year: int, metrics: dict) -> int:
    """
    Save the output of extract_financial_metrics() into public.financial_metrics.

    metrics keys expected (from FinancialSchema):
        revenue, net_income, operating_income, cash_flow_from_operations,
        total_assets, total_liabilities, top_risk_factors, top_growth_drivers
    """
    row = {
        "company": company,
        "year": str(year),
        "revenue": str(metrics.get("revenue")) if metrics.get("revenue") is not None else None,
        "net_income": str(metrics.get("net_income")) if metrics.get("net_income") is not None else None,
        "operating_income": str(metrics.get("operating_income")) if metrics.get("operating_income") is not None else None,
        "cash_flow": str(metrics.get("cash_flow_from_operations")) if metrics.get("cash_flow_from_operations") is not None else None,
        "total_assets": str(metrics.get("total_assets")) if metrics.get("total_assets") is not None else None,
        "total_liabilities": str(metrics.get("total_liabilities")) if metrics.get("total_liabilities") is not None else None,
        "risk_factors": _stringify_list(metrics.get("top_risk_factors")),
        "growth_drivers": _stringify_list(metrics.get("top_growth_drivers")),
    }

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    try:
        with conn.cursor() as cursor:
            cursor.execute(INSERT_SQL, row)
            new_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Saved metrics for {company} ({year}) with id={new_id}")
        return new_id
    except Exception as e:
        conn.rollback()
        print("Error saving metrics:", e)
        raise
    finally:
        conn.close()