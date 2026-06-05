import pandas as pd
from langchain_core.documents import Document
from pathlib import Path


def load_csv(
    csv_path: str,
    chunk_size: int = 50,
    text_columns: list[str] | None = None,
) -> list[Document]:
    """
    Load a CSV into LangChain Documents using chunked streaming.

    Works for both small and large CSVs — pandas never loads the full
    file into RAM. Instead it reads `chunk_size` rows at a time via an
    iterator, yielding one Document per batch.

    Changes from original:
      - pd.read_csv(chunksize=N) replaces full pd.read_csv()
      - total_rows computed with a fast line-count, not len(df)
      - text_columns filter pushed into pandas (usecols) for efficiency
    """
    csv_path = str(Path(csv_path))

    # Fast line count — no full file load
    with open(csv_path) as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header

    docs: list[Document] = []
    chunk_num = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        usecols=text_columns,   # push column filter into pandas
        low_memory=True,
    )

    for chunk_df in reader:
        chunk_df = chunk_df.dropna(how="all")
        if chunk_df.empty:
            chunk_num += 1
            continue

        headers   = list(chunk_df.columns)
        row_start = chunk_num * chunk_size + 1
        row_end   = row_start + len(chunk_df) - 1

        rows_text = "\n".join(
            " | ".join(f"{col}: {row[col]}" for col in headers)
            for _, row in chunk_df.iterrows()
        )

        chunk_text = f"Columns: {', '.join(headers)}\n\n{rows_text}"

        docs.append(Document(
            page_content=chunk_text,
            metadata={
                "source":     csv_path,
                "doc_type":   "csv",
                "row_start":  row_start,
                "row_end":    row_end,
                "total_rows": total_rows,
                "columns":    ", ".join(headers),
                "chunk_id":   f"csv_chunk_{chunk_num:06d}",
            }
        ))
        chunk_num += 1

    return docs