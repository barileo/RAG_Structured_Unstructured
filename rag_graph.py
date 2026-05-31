from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class RAGState(TypedDict):
    query: str
    retrieved_docs: list[Document]
    answer: str


def make_rag_graph(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Answer the question using only the "
            "provided context. Cite source file and page/row where relevant.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{query}"),
    ])

    def retrieve(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["query"])
        return {**state, "retrieved_docs": docs}

    def generate(state: RAGState) -> RAGState:
        context = "\n\n---\n\n".join(
            f"[{d.metadata.get('source')} | "
            f"page {d.metadata.get('page_number', '')} "
            f"rows {d.metadata.get('row_start', '')}-{d.metadata.get('row_end', '')}]\n"
            f"{d.page_content}"
            for d in state["retrieved_docs"]
        )
        chain = prompt | llm
        answer = chain.invoke({"context": context, "query": state["query"]})
        return {**state, "answer": answer.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()