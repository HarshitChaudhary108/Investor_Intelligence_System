import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv(override=True)

index_name = os.getenv("PINECONE_INDEX_NAME")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not set in environment")

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not set in environment")

pc = Pinecone(api_key=api_key)

def get_index(index_name: str, timeout: int = 60):
    """Returns a ready-to-use Pinecone index handle."""

    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not set in environment")
        index_name = index_name.strip().strip('"').strip("'").lower().replace("_", "-").replace(" ", "-")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    waited = 0
    while not pc.describe_index(index_name).status["ready"]:
        if waited >= timeout:
            raise TimeoutError(f"Index '{index_name}' not ready after {timeout}s")
        time.sleep(1)
        waited += 1

    return pc.Index(index_name)

def upsert_chunks(
    index,
    documents: list,        # LangChain Document objects from SemanticChunker
    embedding_model,        # your EmbeddingModel instance
    source_name: str,
    namespace: str = "",
    batch_size: int = 100,
):
    """Embeds document chunks and upserts them into Pinecone."""
    if not documents:
        print("No documents to upsert.")
        return

    for i in range(0, len(documents), batch_size):
        doc_batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in doc_batch]
        embeddings = embedding_model.embed_documents(texts)

        vectors = []
        for j, (doc, embedding) in enumerate(zip(doc_batch, embeddings)):
            vectors.append({
                "id": f"{source_name}-chunk-{i + j}",
                "values": embedding,
                "metadata": {
                    "text": doc.page_content,
                    "source": source_name,
                    **(doc.metadata or {}),
                },
            })

        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Upserted batch {i // batch_size + 1}: {len(vectors)} vectors")

    print(f"Done. Total vectors upserted: {len(documents)}")