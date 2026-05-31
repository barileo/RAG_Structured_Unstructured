PDF ingestion
The hard part is extraction fidelity. Use PyMuPDF for text-native PDFs and fall back to Tesseract OCR for scanned pages. The real work is in chunking: plain token-based splitting destroys context. Instead, use heading-aware splitting — preserve the document hierarchy (H1 > H2 > body) so each chunk inherits its section path as metadata. A 512-token window with ~64-token overlap prevents cutting mid-sentence across important boundaries. For tables inside PDFs, extract them separately as serialized text (col: value, col: value) rather than letting them get mangled by a naive text extraction.

CSV ingestion
Never embed an entire CSV as a blob. For large files, chunk into row-windows (50–100 rows), and critically — prepend the column headers to every chunk. Without headers, a chunk like "Alice, 82000, Seattle" is meaningless to an embedding model. For wide CSVs with many columns, consider semantic column grouping: cluster related columns together before chunking rows. If queries are mostly analytical ("what's the average salary by department"), consider a hybrid approach — a SQL/pandas query layer for structured queries alongside vector search for freeform questions.


Metadata is non-negotiable
Every chunk should carry: source_file, doc_type (pdf/csv), page_number or row_range, section_heading (for PDFs), and a chunk_id. This lets you filter at retrieval time and cite sources precisely in responses.


Retrieval strategy
Don't rely on dense vector search alone. A hybrid retriever combining dense embeddings with BM25 sparse retrieval handles exact-match queries (product codes, names, IDs from CSVs) that semantic search often misses. Add a cross-encoder re-ranker (e.g. ms-marco-MiniLM) on the top-k results to reorder by relevance before passing to the LLM.


Where LlamaIndex fits
Given your familiarity with LlamaIndex — it handles most of this pipeline well. Use SimpleDirectoryReader with custom file extractors per type, SentenceSplitter for PDFs, and a custom CSVReader that prepends headers. The VectorStoreIndex + BM25Retriever combination with a QueryFusionRetriever gets you hybrid retrieval without much boilerplate.
