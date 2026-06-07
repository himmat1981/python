"""
services/document_parser.py

Extract plain text from uploaded documents.
Supports: PDF, DOCX, TXT, MD

Usage:
    text = parse_document("report.pdf", file_bytes)
"""

import io


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from all pages of a PDF."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract paragraph text from a DOCX file."""
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a plain-text or Markdown file as UTF-8."""
    return file_bytes.decode("utf-8", errors="replace")


def parse_document(filename: str, file_bytes: bytes) -> str:
    """
    Extract text from an uploaded document.

    Args:
        filename:   Original filename — used to determine type by extension.
        file_bytes: Raw file content.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
