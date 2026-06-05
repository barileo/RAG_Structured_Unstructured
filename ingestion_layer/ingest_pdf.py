import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
from pathlib import Path
from langchain_core.documents import Document


def extract_text_pymupdf(page) -> str:
    """Primary extraction — fast, preserves layout."""
    return page.get_text("text").strip()


def extract_text_ocr(page) -> str:
    """Fallback for scanned pages — rasterize then OCR."""
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img).strip()


def _table_to_rows(table: list[list]) -> list[list[str]]:
    """Normalise a raw pdfplumber table to clean string rows."""
    return [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in table
        if any(cell for cell in row)          # drop fully-empty rows
    ]


def _rows_match_header(rows: list[list[str]], header: list[str]) -> bool:
    """
    Detect whether `rows` is a continuation fragment (no header) or a new table.

    A fragment is recognised when:
      - The first row does NOT match the known header exactly, AND
      - The column count matches the header column count.

    If the first row IS the header repeated (some PDFs do this), we strip it.
    """
    if not rows or not header:
        return False
    return len(rows[0]) == len(header)


def _is_repeated_header(row: list[str], header: list[str]) -> bool:
    """True if this row is just the header repeated — strip it."""
    return [c.lower().strip() for c in row] == [c.lower().strip() for c in header]


def _rows_to_text(header: list[str], data_rows: list[list[str]]) -> str:
    """Serialise header + rows into LLM-friendly key:value format."""
    lines = []
    for row in data_rows:
        pairs = " | ".join(
            f"{h}: {v}" for h, v in zip(header, row) if h
        )
        lines.append(pairs)
    return "\n".join(lines)


# ── multi-page table extractor ────────────────────────────────────────────────

def extract_tables_multipage(pdf_path: str) -> list[Document]:
    """
    Extract all tables from a PDF, correctly stitching fragments that
    span page boundaries into single Documents.

    Algorithm
    ---------
    State machine with two states: IDLE and IN_TABLE.

    IDLE      — scanning pages for a table start.
                When found, capture the header row, buffer the data rows,
                note the start page, and transition → IN_TABLE.

    IN_TABLE  — on each subsequent page, check whether the FIRST table on
                that page is a continuation (same column count, no matching
                header as first row).
                  • Continuation → append its rows to the buffer.
                  • New table / end of table → flush the buffer as a Document,
                    start fresh with the new table.
    """
    docs: list[Document] = []

    # Pending (possibly still growing) table
    pending_header:     list[str]       = []
    pending_rows:       list[list[str]] = []
    pending_start_page: int             = 0
    pending_end_page:   int             = 0
    in_table:           bool            = False

    def flush(end_page: int):
        nonlocal in_table, pending_header, pending_rows
        if not pending_header or not pending_rows:
            in_table = False
            return
        content = (
            f"Table (pages {pending_start_page}–{end_page})\n"
            f"Columns: {', '.join(pending_header)}\n\n"
            + _rows_to_text(pending_header, pending_rows)
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source":       pdf_path,
                "doc_type":     "pdf_table",
                "page_start":   pending_start_page,
                "page_end":     end_page,
                "column_count": len(pending_header),
                "row_count":    len(pending_rows),
            }
        ))
        pending_header.clear()
        pending_rows.clear()
        in_table = False

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                # No table on this page → flush any open table
                if in_table:
                    flush(pending_end_page)
                continue

            for t_idx, raw_table in enumerate(tables):
                rows = _table_to_rows(raw_table)
                if not rows:
                    continue

                if not in_table:
                    # ── Start a new table ──────────────────────────────────
                    pending_header     = rows[0]
                    pending_rows       = rows[1:]
                    pending_start_page = page_num
                    pending_end_page   = page_num
                    in_table           = True

                else:
                    # ── We're mid-table: continuation or new table? ────────
                    if (
                        t_idx == 0                                      # first table on page
                        and _rows_match_header(rows, pending_header)    # same column count
                        and not _is_repeated_header(rows[0], pending_header)  # not a re-printed header
                    ):
                        # Continuation fragment — append rows
                        pending_rows.extend(rows)
                        pending_end_page = page_num

                    elif (
                        t_idx == 0
                        and _rows_match_header(rows, pending_header)
                        and _is_repeated_header(rows[0], pending_header)
                    ):
                        # Header repeated at top of continuation page — skip it, append rest
                        pending_rows.extend(rows[1:])
                        pending_end_page = page_num

                    else:
                        # Different table — flush current, start new
                        flush(pending_end_page)
                        pending_header     = rows[0]
                        pending_rows       = rows[1:]
                        pending_start_page = page_num
                        pending_end_page   = page_num
                        in_table           = True

        # End of PDF — flush any remaining open table
        if in_table:
            flush(pending_end_page)

    return docs






# def extract_tables_from_page(pdf_path: str, page_num: int) -> list[str]:
#     """Extract tables from a page as serialized key-value strings."""
#     tables_text = []
#     with pdfplumber.open(pdf_path) as pdf:
#         page = pdf.pages[page_num]
#         for table in page.extract_tables():
#             if not table:
#                 continue
#             headers = table[0]
#             for row in table[1:]:
#                 # "col: value, col: value" format — LLM-friendly
#                 row_str = ", ".join(
#                     f"{h}: {v}" for h, v in zip(headers, row)
#                     if h and v
#                 )
#                 tables_text.append(row_str)
#     return tables_text


# ── main PDF loader ───────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> list[Document]:
    """
    Load a PDF into LangChain Documents.
    - Text pages: heading-aware semantic extraction
    - Image pages: OCR fallback (Tesseract)
    - Tables: multi-page stitching via extract_tables_multipage()
    """
    pdf_path_str = str(Path(pdf_path))
    docs: list[Document] = []
    current_section = "Introduction"

    # ── Text extraction ───────────────────────────────────────────────────
    with fitz.open(pdf_path_str) as pdf:
        for page_num, page in enumerate(pdf):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["size"] >= 14:
                            current_section = span["text"].strip()

            text = extract_text_pymupdf(page)

            if len(text) < 100:
                print(f"  [OCR] Page {page_num+1}: {len(text)} chars from PyMuPDF → Tesseract")
                text = extract_text_ocr(page)

            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source":      pdf_path_str,
                        "doc_type":    "pdf",
                        "page_number": page_num + 1,
                        "section":     current_section,
                        "total_pages": len(pdf),
                    }
                ))

    # ── Multi-page table extraction ───────────────────────────────────────
    table_docs = extract_tables_multipage(pdf_path_str)
    print(f"  [tables] Extracted {len(table_docs)} table(s) "
          f"({sum(d.metadata['row_count'] for d in table_docs)} total rows)")
    docs.extend(table_docs)

    return docs