from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# from langchain.
from langchain_core.documents import Document


def build_retriever(docs: list[Document]) -> EnsembleRetriever:
    """
    Hybrid retriever: dense (FAISS) + sparse (BM25).
    - Dense handles semantic queries
    - BM25 handles exact matches (IDs, names, codes in CSVs)
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Dense vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    dense_retriever = vector_store.as_retriever(
        search_type="mmr",          # diversity via max marginal relevance
        search_kwargs={"k": 8, "fetch_k": 20},
    )

    # Sparse BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 8

    # Ensemble: equal weight — tune based on your query mix
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )