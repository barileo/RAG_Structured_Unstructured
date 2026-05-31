import pandas as pd
from langchain_core.documents import Document


def load_csv(
    csv_path: str,
    chunk_size: int = 50,
    text_columns: list[str] | None = None,
) -> list[Document]:
    """
    Load a CSV into LangChain Documents.
    - Column headers prepended to every chunk (critical for embedding quality)
    - Row-window chunking with configurable batch size
    - Optional: restrict to specific text columns for embedding
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")  # drop completely empty rows

    # Use subset of columns if specified
    if text_columns:
        df = df[text_columns]

    headers = list(df.columns)
    docs: list[Document] = []

    for start in range(0, len(df), chunk_size):
        chunk_df = df.iloc[start: start + chunk_size]

        # Serialize each row as "col: value | col: value | ..."
        rows_text = "\n".join(
            " | ".join(f"{col}: {row[col]}" for col in headers)
            for _, row in chunk_df.iterrows()
        )

        # Prepend headers so every chunk is self-contained
        chunk_text = f"Columns: {', '.join(headers)}\n\n{rows_text}"

        docs.append(Document(
            page_content=chunk_text,
            metadata={
                "source": csv_path,
                "doc_type": "csv",
                "row_start": start + 1,
                "row_end": start + len(chunk_df),
                "total_rows": len(df),
                "columns": ", ".join(headers),
            }
        ))

    return docs