"""
Run from the project root (same folder as Ingestion/, rag/, routes/, vector_store/):
    uvicorn main:app --reload
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from Ingestion.ingest2vectorstore import ingest_documents, index_name
from vector_store.vectorDB import get_index
from rag.kpi_extractor import extract_financial_metrics
from schemas.chatSchema import ChatSchema
from llm.LLM import groq_llm
from vector_store.retriever import get_retriever
import asyncio

app = FastAPI(title="Investor Intelligence System API")

RAW_PDF_DIR = Path("data/raw_pdf")
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

# Mount your existing /chat route unchanged


# ---------------------------------------------------------------------
# 1. Ingestion: save PDF -> chunk -> upsert into Pinecone
# ---------------------------------------------------------------------
@app.post("/ingest")
async def ingest_pdf(company: str, year: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        dest_path = RAW_PDF_DIR / file.filename
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        index = await get_index(index_name=index_name)
        ingest_documents(pdf_file=str(dest_path), index=index)

        return {
            "status": "success",
            "company": company,
            "year": year,
            "path": str(dest_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# 2. Metrics: RAG-based KPI extraction
# ---------------------------------------------------------------------
@app.get("/metrics/{company}/{year}")
async def get_metrics(company: str, year: str):
    try:
        metrics = await extract_financial_metrics(company=company, year=int(year))
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics could be extracted.")
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# 3. /chat comes from routes.chat via include_router() above — no changes needed
# ---------------------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatSchema):
    try:
            search_query = f"{request.company} {request.year} {request.inquiry}"
            retriever = get_retriever()
            docs = await retriever.ainvoke(search_query)
            context = "\n\n".join(doc.page_content for doc in docs[:2])
    
            prompt = f"""
            Use the following context from corporate reports to answer the user's question.
            If the context does not contain relevant information,
            politely indicate that you did not have enough data.
    
            Company: {request.company}
            Year: {request.year}
    
            Context:
            {context}
    
            User Question: {request.inquiry}
    
            Answer:
            """
    
            results = await groq_llm.ainvoke(prompt)
            output = results.content
            return {"answer": output}
    
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)