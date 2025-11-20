import pathlib
from typing import List

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(pdf_path: str) -> str:
    """Extract raw text from all pages of a PDF."""
    path = pathlib.Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            normalized = page_text.strip()
            if normalized:
                pages.append(normalized)
            else:
                print(f"[!] Page {idx} contained no extractable text.")

    combined = "\n\n".join(pages).strip()
    if not combined:
        raise ValueError("No text extracted from the PDF.")
    return combined


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
) -> List[str]:
    """Split long text into overlapping chunks for embeddings."""
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

