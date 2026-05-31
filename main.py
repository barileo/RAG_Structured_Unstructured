from dotenv import load_dotenv
load_dotenv()

from ingest_pdf import load_pdf
from ingest_csv import load_csv
from chunker import chunk_document
from retriever import build_retriever
from rag_graph import make_rag_graph

def main():
    print("Hello from rag-structured-unstructured!")
    # 1. Load
    # pdf_docs = (load_pdf("data/sample_report.pdf")+load_pdf("data/sample_report.pdf"))
    # pdf_docs = load_pdf("data/sample_report.pdf")
    pdf_docs = load_pdf("data/scanned_sample.pdf")

    csv_docs = load_csv("data/sample_data.csv", chunk_size=50)
    all_docs = pdf_docs + csv_docs
    print(f"Loaded {len(all_docs)} raw documents")

    # 2. Chunk
    chunks = chunk_document(all_docs)
    print(f"Created {len(chunks)} chunks")

    # 3. Build retriever
    retriever = build_retriever(chunks)

    # 4. Build LangGraph pipeline
    rag = make_rag_graph(retriever)

    # 5. Query
    # result = rag.invoke({"query": "What were the key findings in Q3?"})
    # result = rag.invoke({"query": "PgBouncer in front of"})

    # Now I want to give it multiple tasks to do with temperatur 0 and see what happens
    result = rag.invoke({"query": "PgBouncer in front of? and What is 2 + 2"})
    print("\n=== Answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
