import os
from vector_store.vectorDB import upsert_chunks, get_index
from .pdf_to_markdown import PDF2Markdown
from .semantic_chunker import chunk_markdown
from .embedding_model import EmbeddingModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

index_name = os.getenv("PINECONE_INDEX_NAME")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not set in environment")

converter_obj = PDF2Markdown()
embeddingmodel = EmbeddingModel()

def ingest_documents(pdf_file: str, index):
    """
    ingesting the documents to the vector db
    """
    pdf_file = Path(pdf_file)
    source = pdf_file.stem

    md_file = converter_obj.convert_pdf(
        pdf_file= pdf_file,
        directory= "data/markdown"
    )
    if not md_file:
        raise ValueError(f"PDF to markdown conversion failed for {pdf_file}")

    chunks = chunk_markdown(
        embedding_model= embeddingmodel,
        markdown_file= md_file
    )
    if not chunks:
        raise ValueError(f"No chunks produced for {pdf_file}")

    upsert_chunks(
        index= index,
        documents= chunks,
        embedding_model= embeddingmodel,
        source_name= source
    )


def ingest_directory(input_dir: str):
    """
    Ingest the directory of relevant pdfs to the vector store
    """
    input_dir = Path(input_dir)
    print(f"input directory accessed")

    index = get_index(index_name=index_name)

    for pdf_file in input_dir.glob("*.pdf"):
        try:
            print(f"File: {Path(pdf_file).stem}")
            ingest_documents(pdf_file=pdf_file, index=index)
        except Exception as e:
            print(f"failed on {pdf_file.stem}: {e}")
            continue

if __name__ == "__main__":
    input_dir = "data/raw_pdf"
    ingest_directory(input_dir=input_dir)