from sentence_transformers import SentenceTransformer
from pathlib import Path
from langchain_core.embeddings import Embeddings

class EmbeddingModel(Embeddings):
    """Create embeddings for text chunks using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace sentence-transformers model.
                Good defaults:
                - "all-MiniLM-L6-v2"      -> fast, 384 dims, solid baseline
                - "all-mpnet-base-v2"     -> slower, 768 dims, higher quality
                - "BAAI/bge-small-en-v1.5" -> strong for retrieval/RAG
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 32):
        """
        Embed a list of text chunks.

        Args:
            texts: List of strings (e.g. markdown chunks)
            batch_size: Number of texts encoded per batch

        Return:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # important for cosine similarity search
        )
        return embeddings.tolist()

    def embed_query(self, query: str):
        """Embed a single query string (e.g. user's search question)"""
        embeddings = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embeddings.tolist()


if __name__ == "__main__":
    embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")

    chunks = [
        "Apple reported record Q4 revenue driven by iPhone sales.",
        "The company's services segment grew 12% year over year.",
        "Gross margin improved due to a favorable product mix.",
    ]

    embeddings = embedder.embed_texts(chunks)
    print(f"Embeddings shape: {embeddings.shape}")