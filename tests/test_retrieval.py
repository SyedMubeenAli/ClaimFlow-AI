from decimal import Decimal

from claimflow.extract import extract_pdf_text, has_text, split_pages
from claimflow.money import amount_appears, format_amount, parse_amount
from claimflow.retrieval import BM25Retriever, chunk_pages
from tests.helpers import POLICY_PAGES, make_pdf


def policy_chunks():
    text = extract_pdf_text(make_pdf(POLICY_PAGES))
    return chunk_pages("d1", "policy.pdf", split_pages(text))


# ---- extraction

def test_extraction_keeps_text_and_page_numbers():
    pages = split_pages(extract_pdf_text(make_pdf(POLICY_PAGES)))
    assert [p.number for p in pages] == [1, 2]
    assert "sum insured of 500,000" in pages[0].text and "first 25,000" in pages[1].text


def test_damaged_and_blank_pdfs_are_told_apart():
    assert extract_pdf_text(b"%PDF-1.4 this is not really a pdf") is None
    blank = extract_pdf_text(make_pdf([[]]))
    assert blank is not None and not has_text(blank)
    assert has_text(extract_pdf_text(make_pdf([["hello"]])))


# ---- money

def test_amounts_are_parsed_from_common_formats():
    assert parse_amount("Rs. 1,250.00/-") == Decimal("1250.00")
    assert parse_amount("PKR 5,00,000") == Decimal("500000")
    assert parse_amount(40000) == Decimal("40000")
    assert parse_amount("none") is None and parse_amount(None) is None and parse_amount(True) is None
    assert format_amount(Decimal("500")) == "500.00"


def test_an_amount_must_be_in_the_text_to_be_trusted():
    assert amount_appears(Decimal("25000"), "You must pay the first 25,000 of every claim.")
    assert amount_appears(Decimal("1250"), "Total: 1250.00 only")
    assert not amount_appears(Decimal("9999"), "Total 1,250 and 40,000")
    assert not amount_appears(Decimal("250"), "Total 1,250")        # a substring is not a match


# ---- chunking and search

def test_chunks_carry_source_page_and_stay_short():
    chunks = policy_chunks()
    assert all(c.source == "policy.pdf" and len(c.text) <= 700 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert len({c.id for c in chunks}) == len(chunks)


def test_a_chunk_never_crosses_a_page():
    for chunk in policy_chunks():
        assert ("Excess" in chunk.text) == (chunk.page == 2 and "Excess" in chunk.text)
    assert not any("Collision cover" in c.text and "Excess" in c.text for c in policy_chunks())


def test_long_text_is_split_without_losing_anything():
    from claimflow.extract import Page
    long_text = "word " * 400
    chunks = chunk_pages("d9", "x.pdf", [Page(1, long_text)], max_chars=300)
    assert len(chunks) > 1 and all(len(c.text) <= 300 for c in chunks)
    assert sum(c.text.count("word") for c in chunks) == 400


def test_search_finds_the_right_clause():
    retriever = BM25Retriever(policy_chunks())
    assert "25,000" in retriever.search("deductible excess amount you pay", 1)[0][0].text
    assert "racing" in retriever.search("exclusions what is not covered", 1)[0][0].text
    assert "500,000" in retriever.search("sum insured limit of liability", 1)[0][0].text
    assert retriever.search("zebra quantum", 3) == []


def test_search_matches_plurals_and_combines_queries_without_duplicates():
    retriever = BM25Retriever(policy_chunks())
    assert retriever.search("exclusion", 1)[0][0].page == 2
    found = retriever.search_many(["excess", "excess amount", "exclusions", "sum insured"], per_query=3, total=10)
    assert len({c.id for c in found}) == len(found)
    assert [c.page for c in found] == sorted(c.page for c in found)    # reading order
