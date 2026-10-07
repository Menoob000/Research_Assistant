import os
from typing import List, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

from config import EMBED_MODEL_ID

load_dotenv()


class VectorStore:
    def __init__(self, collection_name: str = "Research_assistant"):
        self.collection_name = collection_name
        self.client = QdrantClient(
            location=os.environ.get("QDRANT_ENDPOINT"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )

        # Dense embeddings using sentence-transformers model via OpenRouter
        # check_embedding_ctx_length=False is required for custom model endpoints
        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL_ID,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
        )

        # Sparse embeddings for BM25 hybrid search
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Ensure the collection exists with both dense and sparse configurations
        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def _ensure_collection(self):
        """Creates the Qdrant collection if it does not already exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Dimension for sentence-transformers/all-MiniLM-L6-v2
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    "langchain-sparse": models.SparseVectorParams()
                },
            )
            print(f"Created hybrid Qdrant collection: {self.collection_name}")

    def add_documents(self, docs: List[Document]) -> List[str]:
        """Ingests a list of Document objects into the vector store."""
        if not docs:
            print("No documents provided to index.")
            return []
        ids = self.vector_store.add_documents(docs)
        print(f"Successfully indexed {len(ids)} document chunks.")
        return ids

    def index_from_loader(self, input_path: str = "./data") -> List[str]:
        """Loads and chunks documents from input_path, then indexes them."""
        from rag.doc_loader import load_files
        docs = load_files(input_path)
        return self.add_documents(docs)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Performs hybrid retrieval for the given query and returns top-k documents."""
        return self.vector_store.similarity_search(query=query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Performs hybrid retrieval and returns documents along with similarity scores."""
        return self.vector_store.similarity_search_with_score(query=query, k=k)

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        """Returns a LangChain retriever suitable for use with agents or tools."""
        return self.vector_store.as_retriever(**kwargs)


def main():
    store = VectorStore()
    print("VectorStore initialized.")

    doc_ids = store.index_from_loader('./data')
    print(f"Indexed document IDs: {doc_ids[:3]}")
    # 2. Search
    results = store.retrieve("air pollution monitoring", k=2)

    for idx, doc in enumerate(results):
        print(f"\nResult {idx + 1}:")
        print(doc.page_content[:150], "...")


if __name__ == "__main__":
    main()