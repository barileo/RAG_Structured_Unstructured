"""
rag_graph.py
============
LangGraph pipeline with a query router.

                    ┌─────────────┐
                    │ route_query │
                    └──────┬──────┘
              vector ◄─────┴─────► graph
                 │                    │
        ┌────────▼────────┐  ┌────────▼────────┐
        │ retrieve_vector │  │  retrieve_graph  │
        └────────┬────────┘  └────────┬────────┘
                 └─────────┬──────────┘
                    ┌──────▼──────┐
                    │   generate  │
                    └─────────────┘

Router classification
---------------------
  vector : freeform search, PDF content, specific record lookup,
           "show me", "find", "what does the report say"
  graph  : relational / aggregation over employee+expense data,
           "who", "which department", "total", "highest", "relationship"
"""

from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from orchestration_layer.graph_rag_chain import make_graph_chain


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query:          str
    route:          Literal["vector", "graph"] | None
    retrieved_docs: list[Document]
    answer:         str


# ── Graph factory ─────────────────────────────────────────────────────────────

def make_rag_graph(retriever, graph_retriever=None):
    """
    Build the LangGraph RAG pipeline.

    Parameters
    ----------
    retriever       : EnsembleRetriever (FAISS + BM25) from retriever.py
    graph_retriever : GraphRetriever from graph_retriever.py
                      If None, all queries route to vector path.
    """
    llm          = ChatOpenAI(model="gpt-4o", temperature=0)
    graph_chain  = make_graph_chain()
    use_graph    = graph_retriever is not None

    # ── Router prompt ──────────────────────────────────────────────────────
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Classify the user question into exactly one category:\n\n"
            "  vector — PDF content, general knowledge, freeform search,\n"
            "           specific document lookup, incident reports, business metrics\n\n"
            "  graph  — questions about employee–expense relationships,\n"
            "           'who spent', 'which department', 'total expenses',\n"
            "           'highest spender', 'how many employees', manager queries,\n"
            "           any question needing a JOIN between people and money\n\n"
            "Reply with ONLY one word: vector or graph."
        )),
        ("human", "{query}"),
    ])
    router_chain = router_prompt | llm

    # ── Generation prompt (vector path) ────────────────────────────────────
    vector_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Answer the question using only the "
            "provided context. Cite source file and page/row where relevant.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{query}"),
    ])

    # ── Nodes ──────────────────────────────────────────────────────────────

    def route_query(state: RAGState) -> RAGState:
        if not use_graph:
            return {**state, "route": "vector"}
        result = router_chain.invoke({"query": state["query"]})
        route  = result.content.strip().lower()
        if route not in ("vector", "graph"):
            route = "vector"
        print(f"  [router] → {route}")
        return {**state, "route": route}

    def retrieve_vector(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["query"])
        return {**state, "retrieved_docs": docs}

    def retrieve_graph(state: RAGState) -> RAGState:
        docs = graph_retriever.invoke(state["query"])
        return {**state, "retrieved_docs": docs}

    def generate(state: RAGState) -> RAGState:
        docs  = state["retrieved_docs"]
        query = state["query"]

        if state["route"] == "graph":
            answer = graph_chain(docs, query)
        else:
            context = "\n\n---\n\n".join(
                f"[{d.metadata.get('source', '?')} | "
                f"page {d.metadata.get('page_number', '')} "
                f"rows {d.metadata.get('row_start', '')}-{d.metadata.get('row_end', '')}]\n"
                f"{d.page_content}"
                for d in docs
            )
            answer = (vector_prompt | llm).invoke({
                "context": context,
                "query":   query,
            }).content

        return {**state, "answer": answer}

    def pick_route(state: RAGState) -> str:
        return state["route"]

    # ── Build graph ────────────────────────────────────────────────────────
    graph = StateGraph(RAGState)

    graph.add_node("route_query",     route_query)
    graph.add_node("retrieve_vector", retrieve_vector)
    graph.add_node("retrieve_graph",  retrieve_graph)
    graph.add_node("generate",        generate)

    graph.set_entry_point("route_query")

    graph.add_conditional_edges(
        "route_query",
        pick_route,
        {
            "vector": "retrieve_vector",
            "graph":  "retrieve_graph",
        }
    )

    graph.add_edge("retrieve_vector", "generate")
    graph.add_edge("retrieve_graph",  "generate")
    graph.add_edge("generate",        END)

    return graph.compile()