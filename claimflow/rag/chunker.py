"""
Splits a policy into chunks that follow its structure.

A chunk never crosses a page (so a citation names one page) and never crosses a
heading (so a clause stays under its own section title). The section title travels
with every chunk as metadata, including when a long section continues on the next
page. The chunk text itself is the page's own words with whitespace collapsed, so a
quote can be checked against it exactly.

Headings are recognised by shape: "Section 4. Excess", "4.2 Exclusions" or
"GENERAL CONDITIONS". Policies that use another style still work. Their chunks just
carry no section title and are split by page and length only.
"""

from __future__ import annotations

import re

from ..extract import Page
from ..retrieval import Chunk

MAX_CHARS = 800

_SECTION = re.compile(r"^(?:section|part|article|clause|chapter)\s+[A-Za-z0-9.\-]+(?:\s*[:.\-]\s*\S.{0,70})?\.?$", re.I)
_NUMBERED = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,2}[.)]?\s+[A-Z][A-Za-z ,&/'()-]{2,70}$")
_CAPS = re.compile(r"^[A-Z][A-Z &/,'()-]{3,60}$")


def is_heading(line: str) -> bool:
    return len(line) <= 90 and bool(_SECTION.match(line) or _NUMBERED.match(line) or _CAPS.match(line))


def clean_heading(line: str) -> str:
    return line.strip().rstrip(".").strip()


def _is_parent(line: str) -> bool:
    """A top-level heading ("Section 4. Excess" or "EXCLUSIONS"), as opposed to a numbered sub-clause."""
    return bool(_SECTION.match(line) or _CAPS.match(line))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;:!?])\s+", text) if s]


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    """Join sentences into pieces of at most `max_chars`, splitting a single huge sentence if it must."""
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        parts = [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)] or [""]
        for part in parts:
            if buffer and len(buffer) + 1 + len(part) > max_chars:
                pieces.append(buffer)
                buffer = ""
            buffer = f"{buffer} {part}".strip()
    if buffer:
        pieces.append(buffer)
    return pieces


def chunk_policy(doc_ref: str, source: str, pages: list[Page], max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""      # the label carried on each chunk, for example "Section 6. Making a claim / 6.1 Telling us"
    parent = ""       # the latest top-level heading
    for page in pages:
        count = 0
        runs: list[tuple[str, list[str]]] = [(section, [])]   # (section title, lines under it)
        for raw in page.text.splitlines():
            line = " ".join(raw.split())
            if not line:
                continue
            if is_heading(line):
                title = clean_heading(line)
                if _is_parent(line):
                    parent = section = title
                else:
                    section = f"{parent} / {title}" if parent else title
                previous = runs[-1][1]
                if previous and all(is_heading(l) for l in previous):
                    # A heading directly under another heading ("Section 2" then "2.1 Collision"):
                    # keep them together instead of leaving a chunk that is only a title.
                    runs[-1] = (section, previous + [line])
                else:
                    runs.append((section, [line]))
            else:
                runs[-1][1].append(line)
        for title, lines in runs:
            if not lines:
                continue
            for text in _pack(_sentences(" ".join(lines)), max_chars):
                chunks.append(Chunk(f"{doc_ref}-p{page.number}-{count}", source, page.number, text, title, doc_ref))
                count += 1
    return chunks
