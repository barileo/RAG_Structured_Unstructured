"""
graph_retriever.py
==================
Retrieves relevant subgraph context for a given query.

Strategy
--------
1. Extract keywords from the query (employee names, departments,
   categories, amounts, project codes).
2. Find matching nodes in the NetworkX graph via case-insensitive
   substring search.
3. For each matched node, pull its immediate neighbourhood
   (depth=1 by default — the node + all its direct edges + neighbours).
4. Serialise the subgraph as structured text the LLM can reason over.
5. Optionally: run a secondary vector search over GraphDocuments
   (the LLM-extracted entity summaries) for semantic fallback.

The output is a list[Document] — same interface as EnsembleRetriever —
so it plugs directly into the existing rag_graph.py retrieve node.
"""

from __future__ import annotations

import re
import networkx as nx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class GraphRetriever:
    """
    Retrieves subgraph context from a NetworkX graph.

    Parameters
    ----------
    graph       : NetworkX graph built by ingest_graph.build_graph()
    graph_docs  : GraphDocuments from the same build — used for vector fallback
    depth       : neighbourhood traversal depth (1 = immediate neighbours)
    top_k       : max nodes to match per query keyword
    use_vector_fallback : if True, build a FAISS index over graph doc summaries
    """

    def __init__(
        self,
        graph: nx.Graph,
        graph_docs: list,
        depth: int = 1,
        top_k: int = 5,
        use_vector_fallback: bool = True,
    ):
        self.graph   = graph
        self.depth   = depth
        self.top_k   = top_k

        # Build node lookup: lowercase node id → real node id
        self._node_index = {
            str(n).lower(): n for n in graph.nodes()
        }

        # Optional FAISS vector index over entity summary texts
        self._vector_store = None
        if use_vector_fallback and graph_docs:
            self._build_vector_fallback(graph_docs)

    # ── vector fallback ───────────────────────────────────────────────────────

    def _build_vector_fallback(self, graph_docs: list):
        """
        Build a small FAISS index over GraphDocument source texts.
        Used when keyword lookup finds no matching nodes.
        """
        summaries = []
        for gdoc in graph_docs:
            text = gdoc.source.page_content
            summaries.append(Document(
                page_content=text,
                metadata={"doc_type": "graph_entity_summary"}
            ))
        if summaries:
            # Large embedding costly
            # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

            # Small embeddins
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._vector_store = FAISS.from_documents(summaries, embeddings)
            print(f"  [graph_retriever] Vector fallback: {len(summaries)} entity summaries indexed")

    # ── keyword extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """
        Extract candidate node keywords from the query.
        Captures: EMP-XXXXX, EXP-XXXXXX, PRJ-XXX IDs,
                  capitalised words (names, departments),
                  and dollar amounts.
        """
        keywords = []

        # Structured IDs
        keywords += re.findall(r'\b(?:EMP|EXP|PRJ)-\w+\b', query, re.IGNORECASE)

        # Capitalised words and phrases (names, department names)
        keywords += re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)

        # Dollar amounts → might match amount-based node properties
        keywords += re.findall(r'\$[\d,]+', query)

        # Deduplicate, keep longest matches first
        seen = set()
        result = []
        for kw in sorted(keywords, key=len, reverse=True):
            if kw.lower() not in seen:
                seen.add(kw.lower())
                result.append(kw)

        return result

    # ── subgraph serialiser ───────────────────────────────────────────────────

    def _subgraph_to_text(self, nodes: set) -> str:
        """
        Serialise a set of nodes and their edges as structured text.

        Output format:
          Entities:
            - Employee: John Smith (id=EMP-00042, department=Engineering)
          Relationships:
            - John Smith --[MADE_EXPENSE]--> EXP-001234 (amount=4200, status=Approved)
        """
        lines = ["Entities:"]
        visited_nodes = set()

        # Expand to neighbourhood
        expanded = set()
        for node in nodes:
            expanded.add(node)
            for neighbour in nx.neighbors(self.graph, node):
                expanded.add(neighbour)

        for node in expanded:
            data = self.graph.nodes[node]
            node_type = data.get("type", "Entity")
            props = {k: v for k, v in data.items() if k != "type" and v}
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            lines.append(f"  - {node_type}: {node}" + (f" ({prop_str})" if prop_str else ""))
            visited_nodes.add(node)

        lines.append("\nRelationships:")
        seen_edges = set()
        for node in nodes:
            for u, v, edge_data in self.graph.edges(node, data=True):
                edge_key = (min(u, v), max(u, v), edge_data.get("relation", ""))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                rel   = edge_data.get("relation", "RELATED_TO")
                props = {k: v for k, v in edge_data.items()
                         if k != "relation" and v}
                prop_str = f" ({', '.join(f'{k}={v}' for k,v in props.items())})" if props else ""
                lines.append(f"  - {u} --[{rel}]--> {v}{prop_str}")

        return "\n".join(lines)

    # ── main retrieval interface ──────────────────────────────────────────────

    def invoke(self, query: str) -> list[Document]:
        """
        Retrieve relevant subgraph context for a query.
        Returns list[Document] — same interface as EnsembleRetriever.
        """
        keywords = self._extract_keywords(query)
        matched_nodes: set = set()

        # Keyword → node lookup
        for kw in keywords:
            kw_lower = kw.lower()
            for node_lower, node_real in self._node_index.items():
                if kw_lower in node_lower or node_lower in kw_lower:
                    matched_nodes.add(node_real)
                    if len(matched_nodes) >= self.top_k * 3:
                        break

        if matched_nodes:
            context_text = self._subgraph_to_text(matched_nodes)
            return [Document(
                page_content=context_text,
                metadata={
                    "doc_type":       "graph_subgraph",
                    "matched_nodes":  len(matched_nodes),
                    "query_keywords": ", ".join(keywords),
                }
            )]

        # Fallback: vector search over entity summaries
        if self._vector_store:
            print(f"  [graph_retriever] No keyword match — using vector fallback")
            return self._vector_store.similarity_search(query, k=self.top_k)

        return []