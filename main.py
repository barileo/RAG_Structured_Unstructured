from dotenv import load_dotenv
load_dotenv()

from ingestion_layer.ingest_pdf import load_pdf
from ingestion_layer.ingest_csv import load_csv
from ingestion_layer.ingest_graph import build_graph
from embedding_retrival_layer.chunker import chunk_document
from embedding_retrival_layer.retriever import build_retriever
from orchestration_layer.rag_graph import make_rag_graph
from embedding_retrival_layer.graph_retriever import GraphRetriever
from langchain_classic.evaluation import load_evaluator

def main():
    print("Hello from rag-structured-unstructured!")
    print("=== RAG pipeline: PDF + CSV + GraphRAG ===\n")
    
    # ── 1. PDF ingestion (unchanged) ─────────────────────────────────────────
    print("Loading PDF...")
    # pdf_docs = (load_pdf("data/sample_report.pdf")+load_pdf("data/sample_report.pdf"))
    # pdf_docs = load_pdf("data/sample_report.pdf")
    pdf_docs = load_pdf("data/scanned_sample.pdf")

    # ── 2. CSV ingestion — now chunked streaming, works for large files ───────
    print("Loading CSVs (chunked streaming)...")
    small_csv_docs = load_csv("data/sample_data.csv", chunk_size=50)

    # Large relational CSVs — same load_csv(), same interface
    employee_docs  = load_csv("data/employees.csv",  chunk_size=100)
    expense_docs   = load_csv("data/expenses.csv",   chunk_size=100)


    all_docs = pdf_docs + small_csv_docs + employee_docs + expense_docs
    print(f"Loaded {len(all_docs)} raw documents\n")

    # ── 3. Chunk + build vector/BM25 retriever (unchanged) ───────────────────
    print("Chunking and building vector retriever...")
    chunks    = chunk_document(all_docs)
    retriever = build_retriever(chunks)
    print(f"Created {len(chunks)} chunks\n")


    # ── 4. Build knowledge graph from relational CSVs ─────────────────────────
    # Set max_batches to a small number during development to limit LLM calls.
    # Set max_batches=None in production to process all 10k rows.
    print("Building knowledge graph (employees + expenses)...")
    graph, graph_docs = build_graph(
        employees_path="data/employees.csv",
        expenses_path="data/expenses.csv",
        batch_size=10,              # 10 rows ≈ 350 tokens per call
        max_batches=1,              # ← remove or set None for full ingestion
        requests_per_minute=8,      # ← raise if your OpenAI tier allows more
    )
 
    graph_ret = GraphRetriever(
        graph=graph,
        graph_docs=graph_docs,
        depth=1,
        top_k=5,
        use_vector_fallback=True,
    )
    print()

    # ── 5. Build LangGraph pipeline with router ───────────────────────────────
    print("Building LangGraph pipeline...\n")
    rag = make_rag_graph(retriever, graph_retriever=graph_ret)


    # ── 6. Example queries ────────────────────────────────────────────────────
    queries = [
        # Routes → vector (PDF content)
        # "What was the root cause of the October 14 outage?",
 
        # Routes → vector (CSV record lookup)
        # "Show me customers with At Risk status in the Pacific Northwest",
 
        # Routes → graph (relational employee/expense question)
        "Total expenses made by Linda Davis?"

        # Routes → graph (relational employee/expense question)
        # "Which department has the highest total expenses?",
 
        # Routes → graph (person-level relationship)
        # "What expenses did employees in Engineering submit?",
    ]

    # result = rag.invoke({"query": "What were the key findings in Q3?"})
    # result = rag.invoke({"query": "PgBouncer in front of"})

    # Now I want to give it multiple tasks to do with temperatur 0 and see what happens
    # result = rag.invoke({"query": "PgBouncer in front of? and What is 2 + 2"})
    # print("\n=== Answer ===")
    # print(result["answer"])
    for q in queries:
        print(f"{'─' * 60}")
        print(f"Q: {q}")
        result = rag.invoke({"query": q})
        print(f"A: {result['answer']}\n")
        print("Performace of the RAG pipeline---------")
        evaluator = load_evaluator("criteria",criteria="relevance")
        evaluation_result= evaluator.evaluate_strings(input=q,prediction=result['answer'])
        print(evaluation_result)




if __name__ == "__main__":
    main()
