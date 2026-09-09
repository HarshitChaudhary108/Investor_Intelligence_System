import os
from pydantic import BaseModel, Field
from typing import Annotated

class FinancialSchema(BaseModel):
    revenue: str | int | None = Field(description="total revenue")
    net_income: str | int | None = Field(description="net income of the company")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")

