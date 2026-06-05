"""
graph_rag_chain.py
==================
Formats graph-retrieved context for the LLM generation step.

Kept deliberately thin — its only job is to produce a clean, well-labelled
context string from graph subgraph Documents so rag_graph.py stays readable.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


GRAPH_SYSTEM_PROMPT = """\
You are a helpful data analyst with access to a knowledge graph built from
employee and expense records.

The context below contains:
  - Entities (Employees, Departments, Expenses, Categories, Projects)
  - Relationships between them extracted from the data

Answer the question using only the provided graph context.
When relevant, mention specific IDs, amounts, departments, or relationships
you found in the graph. If the graph context does not contain enough
information to answer, say so clearly.

Graph context:
{context}
"""


def format_graph_context(docs: list[Document]) -> str:
    """
    Merge graph subgraph Documents into a single context string
    with clear section labels.
    """
    sections = []
    for i, doc in enumerate(docs, 1):
        doc_type = doc.metadata.get("doc_type", "graph")
        header   = f"[Graph context {i} | type={doc_type}]"
        sections.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def make_graph_chain():
    """
    Returns a callable: (docs, query) -> answer string.

    Used by rag_graph.py in the graph generation node.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", GRAPH_SYSTEM_PROMPT),
        ("human", "{query}"),
    ])

    chain = prompt | llm

    def run(docs: list[Document], query: str) -> str:
        context = format_graph_context(docs)
        return chain.invoke({"context": context, "query": query}).content

    return run