import os
from llm.LLM import groq_llm, invoke_structured
from groq import BadRequestError
from schemas.financial_metrics import FinancialSchema
from vector_store.retriever import get_retriever
import json
from database.savemetrics import save_metrics

Retriever = get_retriever(top_k=2)

def retrieve_context(
        company,
        year,
        retriever= Retriever
    ): 
    """
    Retrieve broad financial context from the vector store.
    """
    query = f"""
    Annual report financial statements,
    income statement,
    balance sheet,
    cash flow statement,
    risks,
    growth drivers,
    financial performance
    for {company} fiscal year {year}
    """

    retrieved_docs = retriever.invoke(query)
    with open("docs.txt", "w", encoding="utf-8") as f:
        for i, doc in enumerate(retrieved_docs):
            f.write(f"--- Doc {i+1} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")

    output = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return output

def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build KPI extraction prompt.
    """
    return f"""
    You are an expert financial analyst.

    Company: {company}
    Year: {year}

    Context:
    {context}

    Extract the following fields and return ONLY a JSON object with these exact keys:
    {{
      "revenue": <number or null>,
      "net_income": <number or null>,
      "operating_income": <number or null>,
      "cash_flow_from_operations": <number or null>,
      "total_assets": <number or null>,
      "total_liabilities": <number or null>,
      "top_risk_factors": [<strings>] or null,
      "top_growth_drivers": [<strings>] or null
    }}

    Rules:
    - Use only the provided context.
    - Return null if a value is unavailable.
    - Financial values must match the report exactly.
    """

def extract_financial_metrics(
    company: str,
    year: int,
    retriever=Retriever
) -> dict:
    """
    Extract KPIs using RAG, with fallback salvage on tool-call formatting failure.
    """
    context = retrieve_context(retriever=retriever, company=company, year=year)

    prompt = build_extraction_prompt(company=company, year=year, context=context)

    structured_llm = groq_llm.with_structured_output(FinancialSchema, method="json_mode")

    try:
        metrics = invoke_structured(structured_llm, prompt)
        metrics = metrics.model_dump()
        save_metrics(company= company, year= year, metrics= metrics)
        
        return metrics
    except BadRequestError as e:
        try:
            failed_json = e.body["error"]["failed_generation"]
            parsed = json.loads(failed_json)
            return FinancialSchema(**parsed).model_dump()
        except Exception:
            raise


def main() -> None:
    year = 2025
    results = extract_financial_metrics(
        company="Apple",
        year=year
    )
    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)

if __name__ == "__main__":
    main()