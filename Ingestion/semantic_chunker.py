from langchain_experimental.text_splitter import SemanticChunker
from pathlib import Path
from .embedding_model import EmbeddingModel

def load(md_file: str):
    """
    load the markdown file
    """
    markdown_file = md_file
    with open(markdown_file, "r", encoding="utf-8") as f:
        content= f.read()
        return content

def chunk_markdown(
        embedding_model,
        markdown_file:str,
    ):
    """
    Semantic chunk the markdown document
    """

    markdown_content = load(md_file=markdown_file)
    splitter = SemanticChunker(
        embeddings= embedding_model,
        breakpoint_threshold_type= "percentile",
    )
    return splitter.create_documents(texts= [markdown_content])


if __name__ == "__main__":
    embed_obj = EmbeddingModel()
    md_file = "data/markdown/2025_Apple.md"
    chunks = chunk_markdown(
        embedding_model= embed_obj,
        markdown_file= md_file
    )
    print(f"Len of Chunks: {len(chunks)}") # 48 Chunks were created