"""Small pure text helpers shared by the agents and the RAG package. No imports from either."""

from __future__ import annotations

import re


def clip(text: object, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def defang(text: object) -> str:
    """
    Untrusted text goes into prompts inside frames such as <document> tags and [CHUNK id]
    headers. Breaking up any '<' and any '[CHUNK' stops that text from writing its own
    closing tag or a fake chunk header and escaping the frame.
    """
    return str(text).replace("<", "< ").replace("[CHUNK", "[ CHUNK")


def str_or_none(value: object, limit: int) -> str | None:
    """A cleaned, length-limited string from a model field, or None for empty and 'null'-like values."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    if not cleaned or cleaned.lower() in {"null", "none", "unknown", "n/a", "not stated"}:
        return None
    return clip(cleaned, limit)


def normalize_quote(text: str) -> str:
    """Lowercase, straight quotes, collapsed whitespace, no leading or trailing ellipsis or quote marks."""
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = " ".join(text.lower().split())
    return re.sub(r"^(?:\.{2,}|\u2026|[\"'\s])+|(?:\.{2,}|\u2026|[\"'\s])+$", "", text)


def plural(count: int, noun: str, many: str | None = None) -> str:
    """'1 reason', '2 reasons', '0 reasons'. Pass `many` for an irregular plural."""
    return f"{count} {noun if count == 1 else (many or noun + 's')}"


def is_are(count: int) -> str:
    return "is" if count == 1 else "are"


def quoted_note(notes: str | None) -> str:
    """
    A reviewer's note as one clean sentence: "The reviewer's note: "Please send the quotation."".
    The note is quoted so it can never run into the next sentence, and ends with a full stop if the
    reviewer did not type one. Returns "" when there is no note.
    """
    text = " ".join((notes or "").split())
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return f'The reviewer\'s note: "{text}"'
