"""Build and query a Pinecone vector store over medical document chunks."""

import os

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
DEFAULT_INDEX_NAME = "medical-chatbot"


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def ensure_index(index_name: str = DEFAULT_INDEX_NAME, cloud: str = "aws", region: str = "us-east-1") -> None:
    """Create the Pinecone index if it doesn't already exist."""
    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if index_name not in [index["name"] for index in client.list_indexes()]:
        client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


def build_vector_store(chunks: list[Document], index_name: str = DEFAULT_INDEX_NAME) -> PineconeVectorStore:
    """Embed chunks and upsert them into the Pinecone index, creating it if needed."""
    ensure_index(index_name)
    return PineconeVectorStore.from_documents(
        documents=chunks, embedding=get_embeddings(), index_name=index_name
    )


def load_vector_store(index_name: str = DEFAULT_INDEX_NAME) -> PineconeVectorStore:
    """Connect to a previously populated Pinecone index without re-embedding."""
    return PineconeVectorStore(index_name=index_name, embedding=get_embeddings())
