"""PDF reading: PyMuPDF first, pypdf as the fallback. The PyMuPDF code is exercised through a stand-in."""

import sys
import types

from claimflow import extract
from claimflow.extract import PAGE_SEPARATOR, extract_pdf_text, extractor_name, read_pdf, render_page_png
from tests.helpers import POLICY_PAGES, SAMPLE_POLICY, Skip, make_pdf


class FakePage:
    def __init__(self, number, text):
        self.number, self.text, self.asked = number, text, []

    def get_text(self, kind, **kwargs):
        self.asked.append((kind, kwargs))
        return self.text

    def get_pixmap(self, dpi):
        outer = self
        return types.SimpleNamespace(tobytes=lambda fmt: f"{fmt}:{outer.number}:{dpi}".encode())


class FakeDocument:
    def __init__(self, texts, metadata=None, needs_pass=False, can_unlock=False):
        self.pages = [FakePage(i, t) for i, t in enumerate(texts)]
        self.page_count, self.metadata = len(texts), metadata or {}
        self.needs_pass, self._unlock, self.closed = needs_pass, can_unlock, False

    def __getitem__(self, i):
        return self.pages[i]

    def authenticate(self, password):
        return self._unlock

    def close(self):
        self.closed = True


def fake_pymupdf(document=None, error=None):
    opened = []
    def open_(stream=None, filetype=None):
        if error:
            raise error
        opened.append((len(stream), filetype))
        return document
    return types.SimpleNamespace(open=open_, opened=opened)


def with_fake(monkey_module, test):
    original = extract._load_pymupdf
    extract._load_pymupdf = lambda: monkey_module
    try:
        return test()
    finally:
        extract._load_pymupdf = original


# ---------------------------------------------------------------- PyMuPDF path (through a stand-in)

def test_pymupdf_is_used_first_and_its_metadata_and_page_numbers_are_kept():
    doc = FakeDocument(["Page one  text", "  Page two\ntext  "], {"title": "My Policy", "author": "A. Writer", "format": "PDF 1.7"})
    module = fake_pymupdf(doc)
    result = with_fake(module, lambda: read_pdf(b"%PDF-1.4 pretend"))
    assert result.extractor == "pymupdf" and result.page_count == 2
    assert [(p.number, p.text) for p in result.pages] == [(1, "Page one text"), (2, "Page two\ntext")]
    assert result.metadata == {"title": "My Policy", "author": "A. Writer"}        # only the short human fields
    assert module.opened == [(len(b"%PDF-1.4 pretend"), "pdf")] and doc.closed
    assert doc.pages[0].asked == [("text", {"sort": True})]                          # reading order, top to bottom


def test_extract_pdf_text_joins_pages_with_a_form_feed():
    module = fake_pymupdf(FakeDocument(["One", "Two"]))
    assert with_fake(module, lambda: extract_pdf_text(b"x")) == f"One{PAGE_SEPARATOR}Two"


def test_pymupdf_failing_falls_back_to_pypdf_and_says_so():
    module = fake_pymupdf(error=RuntimeError("cannot open"))
    result = with_fake(module, lambda: read_pdf(make_pdf(POLICY_PAGES)))
    assert result.extractor == "pypdf" and "sum insured of 500,000" in result.pages[0].text


def test_a_password_protected_pdf_is_not_retried_with_the_second_reader():
    locked = FakeDocument(["secret"], needs_pass=True, can_unlock=False)
    module = fake_pymupdf(locked)
    assert with_fake(module, lambda: read_pdf(make_pdf(POLICY_PAGES))) is None       # the good PDF is ignored: PyMuPDF said locked
    assert locked.closed
    unlocked = FakeDocument(["now readable"], needs_pass=True, can_unlock=True)
    assert with_fake(fake_pymupdf(unlocked), lambda: read_pdf(b"x")).pages[0].text == "now readable"


def test_only_the_first_pages_are_read_and_truncation_is_reported():
    doc = FakeDocument([f"page {i}" for i in range(extract.MAX_PAGES + 5)])
    result = with_fake(fake_pymupdf(doc), lambda: read_pdf(b"x"))
    assert len(result.pages) == extract.MAX_PAGES and result.page_count == extract.MAX_PAGES + 5 and result.truncated


def test_a_page_can_be_rendered_as_an_image_when_pymupdf_is_available():
    module = fake_pymupdf(FakeDocument(["a", "b", "c"]))
    assert with_fake(module, lambda: render_page_png(b"x", 2, dpi=90)) == b"png:1:90"
    assert with_fake(module, lambda: render_page_png(b"x", 0)) is None                # out of range
    assert with_fake(module, lambda: render_page_png(b"x", 4)) is None
    assert with_fake(fake_pymupdf(error=RuntimeError("boom")), lambda: render_page_png(b"x", 1)) is None
    assert with_fake(None, lambda: render_page_png(b"x", 1)) is None                  # PyMuPDF not installed


def test_the_first_reader_is_named_for_display():
    assert with_fake(fake_pymupdf(FakeDocument(["a"])), extractor_name) == "pymupdf"
    assert with_fake(None, extractor_name) == "pypdf"


# ---------------------------------------------------------------- the fallback reader, for real

def test_pypdf_reads_the_sample_policy_with_its_metadata():
    result = with_fake(None, lambda: read_pdf(SAMPLE_POLICY.read_bytes()))
    assert result.extractor == "pypdf" and result.page_count == 6
    assert result.metadata["title"] == "Northgate Mutual Motor Insurance Policy (sample)"
    assert "Section 4. Excess" in result.pages[2].text and "PKR 25,000" in result.pages[2].text


def test_neither_reader_can_open_garbage():
    assert with_fake(None, lambda: read_pdf(b"not a pdf at all")) is None
    assert with_fake(fake_pymupdf(error=ValueError("bad")), lambda: read_pdf(b"not a pdf at all")) is None


# ---------------------------------------------------------------- the real library, when installed

def test_the_real_pymupdf_reads_and_renders_the_sample_policy():
    if extract._load_pymupdf() is None:
        raise Skip("PyMuPDF is not installed on this machine")
    data = SAMPLE_POLICY.read_bytes()
    result = read_pdf(data)
    assert result.extractor == "pymupdf" and result.page_count == 6
    assert "Section 4. Excess" in result.pages[2].text and "PKR 25,000" in result.pages[2].text
    assert render_page_png(data, 3)[:8] == b"\x89PNG\r\n\x1a\n"
