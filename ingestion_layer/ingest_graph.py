"""
ingest_graph.py
===============
Builds a LangChain GraphRAG knowledge graph from two related CSVs.

Flow
----
1. Read employees.csv + expenses.csv via chunked streaming (memory-safe).
2. Join each expense row to its employee record so the LLM sees a
   complete, self-contained unit: who spent what, in which department.
3. Serialise each joined row as a short natural-language sentence.
4. Run LLMGraphTransformer over batches of sentences to extract:
     Nodes  : Employee, Department, Expense, Category, Project
     Edges  : MADE_EXPENSE, BELONGS_TO, IN_CATEGORY, TAGGED_WITH
5. Accumulate all GraphDocuments into a NetworkX in-memory graph.
6. Return the graph + the GraphDocuments list (used by graph_retriever).

Usage
-----
    from ingest_graph import build_graph
    graph, graph_docs = build_graph(
        employees_path="data/employees.csv",
        expenses_path="data/expenses.csv",
        batch_size=50,          # rows per LLM extraction call
        max_batches=None,       # None = process full files; set int for dev
    )
"""

from __future__ import annotations

import time
import pandas as pd
import networkx as nx
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI


# ── helpers ───────────────────────────────────────────────────────────────────

def _row_to_sentence(row: pd.Series) -> str:
    """
    Serialise a joined expense+employee row as a compact sentence.

    Leaner than the original — drops redundant fields (currency, notes
    free-text) that add tokens without helping entity extraction.
    ~35 tokens per row vs ~60 previously.

    Example output:
      "John Smith (EMP-00042, Engineering) submitted EXP-001234:
       Travel $4200 on 2024-03-15, Approved, PRJ-214."
    """
    return (
        f"{row['first_name']} {row['last_name']} "
        f"({row['employee_id']}, {row['department']}) "
        f"submitted {row['expense_id']}: "
        f"{row['category']} ${row['amount']:.0f} "
        f"on {row['expense_date']}, {row['exp_status']}, {row['project_code']}."
    )


# ── main builder ──────────────────────────────────────────────────────────────

def build_graph(
    employees_path: str,
    expenses_path: str,
    batch_size: int = 10,
    max_batches: int | None = None,
    requests_per_minute: int = 8,
) -> tuple[nx.Graph, list]:
    """
    Build a NetworkX knowledge graph from employees + expenses CSVs.

    Parameters
    ----------
    employees_path      : path to employees.csv
    expenses_path       : path to expenses.csv
    batch_size          : rows per LLM extraction call (keep low to stay
                          within TPM limits — 10 rows ≈ 350 tokens/call)
    max_batches         : cap for development/testing (None = full files)
    requests_per_minute : throttle — adds a sleep between batches to avoid
                          429 rate limit errors. Default 8 = ~7.5s between
                          calls, well within a 30k TPM budget at 350 tokens
                          per call (8 × 350 = 2,800 TPM used, leaving headroom).
                          Raise to 20 if you have a higher TPM tier.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Allowed node and relationship types guide the LLM extraction.
    # Being explicit here dramatically improves consistency.
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Employee", "Department", "Expense", "Category", "Project", "Location"],
        allowed_relationships=[
            "MADE_EXPENSE",    # Employee → Expense
            "BELONGS_TO",      # Employee → Department
            "IN_CATEGORY",     # Expense → Category
            "TAGGED_WITH",     # Expense → Project
            "LOCATED_IN",      # Employee → Location
            "APPROVED_BY",     # Expense → Employee (approver)
        ],
        node_properties=["name", "amount", "status", "date", "title", "salary"],
        relationship_properties=["amount", "date", "status"],
    )

    # ── 1. Load employees fully (they're the lookup side of the join) ─────────
    # Even at 10k rows, employee data fits comfortably in memory.
    print("  [graph] Loading employees...")
    emp_df = pd.read_csv(employees_path)
    emp_df = emp_df.rename(columns={"status": "emp_status"})
    print(f"  [graph] {len(emp_df):,} employees loaded")

    # ── 2. Stream expenses + join ─────────────────────────────────────────────
    print("  [graph] Streaming expenses + joining to employees...")
    graph      = nx.Graph()
    graph_docs = []
    batch_num  = 0
    sentences_buf: list[str] = []

    exp_reader = pd.read_csv(expenses_path, chunksize=batch_size)

    for exp_chunk in exp_reader:
        if max_batches and batch_num >= max_batches:
            break

        # Rename status before merge to avoid collision
        exp_chunk = exp_chunk.rename(columns={"status": "exp_status"})

        # expenses.csv also has a 'department' column — drop it before
        # merging so we don't get department_x / department_y suffixes.
        # The authoritative department value always comes from employees.csv.
        exp_chunk = exp_chunk.drop(columns=["department"], errors="ignore")

        # Join expense rows to employee rows on employee_id
        joined = exp_chunk.merge(
            emp_df[["employee_id", "first_name", "last_name",
                    "department", "title", "location", "salary"]],
            on="employee_id",
            how="left",
        )

        # Convert each joined row to a natural-language sentence
        for _, row in joined.iterrows():
            sentences_buf.append(_row_to_sentence(row))

        # Process accumulated sentences as LangChain Documents
        docs_batch = [
            Document(page_content=s) for s in sentences_buf
        ]

        print(f"  [graph] Batch {batch_num + 1}: extracting graph from "
              f"{len(docs_batch)} sentences...")

        try:
            gdocs = transformer.convert_to_graph_documents(docs_batch)
            graph_docs.extend(gdocs)

            # Accumulate nodes and edges into the NetworkX graph
            for gdoc in gdocs:
                for node in gdoc.nodes:
                    graph.add_node(
                        node.id,
                        type=node.type,
                        **{k: v for k, v in (node.properties or {}).items()},
                    )
                for rel in gdoc.relationships:
                    graph.add_edge(
                        rel.source.id,
                        rel.target.id,
                        relation=rel.type,
                        **{k: v for k, v in (rel.properties or {}).items()},
                    )
        except Exception as e:
            print(f"  [graph] Warning: batch {batch_num + 1} extraction failed: {e}")

        sentences_buf.clear()
        batch_num += 1

        # Throttle to stay within requests_per_minute limit.
        # sleep_secs = 60 / RPM, e.g. 8 RPM → sleep 7.5s between calls.
        sleep_secs = 60.0 / requests_per_minute
        print(f"  [graph] Sleeping {sleep_secs:.1f}s (throttle: {requests_per_minute} RPM)...")
        time.sleep(sleep_secs)

    print(f"  [graph] Done — {graph.number_of_nodes():,} nodes, "
          f"{graph.number_of_edges():,} edges, "
          f"{len(graph_docs)} graph documents")

    return graph, graph_docs