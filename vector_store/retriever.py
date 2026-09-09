import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from pydantic import Field

from .vectorDB import get_index
from Ingestion.embedding_model import EmbeddingModel

load_dotenv(override=True)

index_name = os.getenv("PINECONE_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


class PineconeRetriever(BaseRetriever):
    """Custom retriever using the raw Pinecone client (no langchain-pinecone dependency)."""

    index: object = Field(...)
    embedding_model: object = Field(...)
    namespace: str = ""
    top_k: int = 5
    score_threshold: float | None = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_embedding = self.embedding_model.embed_query(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        docs = []
        for match in results.get("matches", []):
            if self.score_threshold is not None and match["score"] < self.score_threshold:
                continue
            metadata = match.get("metadata", {}) or {}
            text = metadata.pop("text", "")
            docs.append(Document(page_content=text, metadata=metadata))

        return docs


def _fetch_docs_for_bm25(index, embedding_model, namespace="", sample_size=200):
    """
    Simple workaround: Pinecone has no native keyword search, so we pull a broad
    sample of chunks via a generic dense query and use that as the BM25 corpus.
    """
    dummy_vector = embedding_model.embed_query(
        "annual report financial statements revenue income balance sheet risks"
    )
    results = index.query(
        vector=dummy_vector,
        top_k=sample_size,
        namespace=namespace,
        include_metadata=True,
    )

    docs = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {}) or {}
        text = metadata.pop("text", "")
        if text:
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


def get_retriever(
    namespace: str = "",
    top_k: int = 5,
    score_threshold: float | None = None,
    use_reranker: bool = True,
):
    index = get_index(index_name=index_name)
    embedding_model = EmbeddingModel()

    # fetch a bit more than needed pre-rerank, so reranker has something to work with
    fetch_k = top_k * 3

    dense_retriever = PineconeRetriever(
        index=index,
        embedding_model=embedding_model,
        namespace=namespace,
        top_k=fetch_k,
        score_threshold=score_threshold,
    )

    bm25_docs = _fetch_docs_for_bm25(index, embedding_model, namespace=namespace)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = fetch_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    if not use_reranker:
        return ensemble_retriever

    compressor = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model="rerank-english-v3.0",
        top_n=top_k,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

if __name__ == "__main__":
    retriever = get_retriever(top_k=5)
    results = retriever.invoke("What was Apple's revenue growth in 2025?")

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])