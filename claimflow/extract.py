"""
PDF reading, done once at upload time and stored with the document.

PyMuPDF is the primary reader. If it is not installed, or it fails on a file, pypdf is
tried instead, and the name of the reader that worked is recorded. Both only read
text and metadata. Nothing inside a PDF is executed.

The text is one string with a form feed between pages, so page numbers survive for
citations. The return value of extract_pdf_text has three meanings:
    None        the file could not be parsed at all (damaged or password protected)
    "" or blank it parsed but has no text layer, which usually means a scan
    text        normal case
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .logging_setup import get_logger

log = get_logger("extract")
logging.getLogger("pypdf").setLevel(logging.ERROR)  # we log our own, clearer message on failure

PAGE_SEPARATOR = "\f"
MAX_PAGES = 60
MAX_CHARS_PER_PAGE = 20_000


@dataclass(frozen=True)
class Page:
    number: int  # starts at 1
    text: str


@dataclass(frozen=True)
class PdfDocument:
    pages: list[Page]
    page_count: int                      # pages in the file, which can exceed len(pages)
    extractor: str                       # "pymupdf" or "pypdf"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.page_count > len(self.pages)


def _clean(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r", "").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _metadata(raw: Any) -> dict[str, str]:
    """Keep only the short, human-readable fields, as plain strings."""
    if not isinstance(raw, dict):
        return {}
    keep = ("title", "author", "subject", "creator", "producer", "creationDate", "creation_date")
    out: dict[str, str] = {}
    for key in keep:
        value = raw.get(key) or raw.get("/" + key.capitalize())
        if value and str(value).strip():
            out[key] = " ".join(str(value).split())[:120]
    return out


# ---------------------------------------------------------------- readers

def _load_pymupdf():
    """The PyMuPDF module, or None. The package is called pymupdf, and older versions import as fitz."""
    try:
        import pymupdf  # type: ignore
        return pymupdf
    except ImportError:
        try:
            import fitz  # type: ignore
            return fitz
        except ImportError:
            return None


def _read_with_pymupdf(data: bytes) -> PdfDocument | None:
    module = _load_pymupdf()
    if module is None:
        return None
    document = module.open(stream=data, filetype="pdf")
    try:
        if getattr(document, "needs_pass", False) and not document.authenticate(""):
            log.warning("PDF is password protected")
            return None
        total = document.page_count
        pages = [Page(i + 1, _clean(document[i].get_text("text", sort=True) or "")[:MAX_CHARS_PER_PAGE])
                 for i in range(min(total, MAX_PAGES))]
        return PdfDocument(pages, total, "pymupdf", _metadata(document.metadata))
    finally:
        document.close()


def _read_with_pypdf(data: bytes) -> PdfDocument | None:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        log.warning("PDF is password protected")
        return None
    total = len(reader.pages)
    pages = [Page(i + 1, _clean(reader.pages[i].extract_text() or "")[:MAX_CHARS_PER_PAGE])
             for i in range(min(total, MAX_PAGES))]
    return PdfDocument(pages, total, "pypdf", _metadata(reader.metadata))


def read_pdf(data: bytes) -> PdfDocument | None:
    """
    Read a PDF with PyMuPDF, falling back to pypdf. None if neither can parse it.

    A file that PyMuPDF opens but finds no text in gets a second try with pypdf, because the two
    readers do not always agree. Only if both find nothing is the file treated as a scan.
    """
    blank: PdfDocument | None = None
    for name, reader in (("pymupdf", _read_with_pymupdf), ("pypdf", _read_with_pypdf)):
        try:
            document = reader(data)
        except Exception as exc:  # both libraries raise many kinds of error on damaged files
            log.warning("%s could not read a PDF: %s", name, type(exc).__name__)
            continue
        if document is None and name == "pymupdf" and _load_pymupdf() is not None:
            return None  # PyMuPDF said "password protected": a second reader will not change that
        if document is None or not document.pages:
            continue
        if any(has_text(p.text) for p in document.pages):
            if document.truncated:
                log.warning("PDF has %d pages; only the first %d were read", document.page_count, MAX_PAGES)
            return document
        blank = blank or document   # opened, but no text: try the other reader before giving up
    return blank


def extractor_name() -> str:
    """Which reader will be used first, for display on the Settings page."""
    return "pymupdf" if _load_pymupdf() is not None else "pypdf"


def extract_pdf_text(data: bytes) -> str | None:
    document = read_pdf(data)
    return None if document is None else PAGE_SEPARATOR.join(p.text for p in document.pages)


def render_page_png(data: bytes, page_number: int, dpi: int = 110) -> bytes | None:
    """The page as a PNG image so a reviewer can see the real page. None if PyMuPDF is unavailable."""
    module = _load_pymupdf()
    if module is None:
        return None
    try:
        document = module.open(stream=data, filetype="pdf")
        try:
            if not 1 <= page_number <= document.page_count:
                return None
            return document[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")
        finally:
            document.close()
    except Exception as exc:
        log.warning("Could not render page %s: %s", page_number, type(exc).__name__)
        return None


def split_pages(text: str) -> list[Page]:
    return [Page(i + 1, t) for i, t in enumerate(text.split(PAGE_SEPARATOR))]


def has_text(text: str | None) -> bool:
    return bool(text and text.replace(PAGE_SEPARATOR, "").strip())
