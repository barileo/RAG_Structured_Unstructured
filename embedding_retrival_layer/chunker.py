from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_document(docs: list[Document])-> list[Document]:
    """
    Chunk PDF docs with overlap. CSV docs are pre-chunked — skip them.
    """

    pdf_docs = [d for d in docs if d.metadata["doc_type"] in ("pdf","pdf_table") ]
    csv_docs = [d for d in docs if d.metadata["doc_type"] =="csv"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=64,separators=["\n\n", "\n", ". ", " "],)

    split_pdf_docs = splitter.split_documents(pdf_docs)

    # Add chunk_id to all docs
    all_docs = split_pdf_docs + csv_docs
    for i, doc in enumerate(all_docs):
        doc.metadata["chunk_id"] = f"chunk_{i:05d}"

    return all_docs
