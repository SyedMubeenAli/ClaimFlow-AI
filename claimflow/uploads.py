"""
Upload handling. Every uploaded file is untrusted until proven otherwise.

Checks, in order: the name is cleaned, the extension is on a short allow-list
and allowed for that kind of document, the size is under the limit, the first
bytes match the extension, and images actually decode. The file is then stored
under a random name, so nothing the uploader typed ever becomes a path.

Files are only ever read as data (text extraction, image bytes). Nothing in an
upload is executed.

This module also holds a small check for text that looks like instructions to
an AI. Documents are treated as evidence, never as commands, so that text is
reported as a risk indicator and never followed.
"""

from __future__ import annotations

import hashlib
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .db import Database
from .errors import UploadError
from .extract import extract_pdf_text
from .logging_setup import get_logger
from .models import DocumentKind, DocumentRecord

log = get_logger("uploads")

ALLOWED_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
SIGNATURES = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# Text kinds must be PDFs so the agents can read them. Photos must be images.
KIND_EXTENSIONS = {
    DocumentKind.POLICY: {".pdf"},
    DocumentKind.ACCIDENT_REPORT: {".pdf"},
    DocumentKind.QUOTATION: {".pdf"},
    DocumentKind.DAMAGE_PHOTO: IMAGE_EXTENSIONS,
    DocumentKind.SUPPORTING: set(ALLOWED_TYPES),
}
MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True)
class ValidatedUpload:
    safe_name: str
    extension: str
    mime: str
    size_bytes: int
    sha256: str


def sanitize_filename(name: str) -> str:
    """Keep only the file name, with safe characters and a sensible length."""
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    if not base:
        return "upload"
    stem, dot, ext = base.rpartition(".")
    if not dot:
        return base[:100]
    return f"{stem[:90]}.{ext[:8]}" if stem else base[:100]


def _check_image(data: bytes) -> None:
    try:
        from PIL import Image  # Pillow ships with Streamlit
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise UploadError("The image is too large (over 40 megapixels).")
            image.verify()
    except UploadError:
        raise
    except Exception as exc:
        raise UploadError("The image file is damaged or is not a real image.") from exc


def validate_upload(filename: str, data: bytes, kind: DocumentKind, max_bytes: int) -> ValidatedUpload:
    safe_name = sanitize_filename(filename)
    extension = Path(safe_name).suffix.lower()
    allowed_here = KIND_EXTENSIONS[kind]

    if extension not in ALLOWED_TYPES:
        raise UploadError(f"'{safe_name}' is not an accepted file type.", hint="Upload a PDF, PNG or JPG.")
    if extension not in allowed_here:
        wanted = ", ".join(sorted(e.lstrip(".").upper() for e in allowed_here))
        raise UploadError(f"A {kind.label.lower()} must be a {wanted} file.")
    if not data:
        raise UploadError(f"'{safe_name}' is empty.")
    if len(data) > max_bytes:
        raise UploadError(f"'{safe_name}' is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if not any(data.startswith(sig) for sig in SIGNATURES[extension]):
        raise UploadError(f"'{safe_name}' does not look like a real {extension.lstrip('.').upper()} file.")
    if extension in IMAGE_EXTENSIONS:
        _check_image(data)

    return ValidatedUpload(safe_name, extension, ALLOWED_TYPES[extension], len(data),
                           hashlib.sha256(data).hexdigest())


def resolve_stored_path(upload_dir: Path, relative: str) -> Path:
    """Turn a stored relative path into a real one, refusing anything outside the upload folder."""
    root = Path(upload_dir).resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise UploadError("That file path is not allowed.")
    return path


def store_document(db: Database, upload_dir: Path, claim_id: int, kind: DocumentKind,
                   filename: str, data: bytes, max_bytes: int) -> DocumentRecord:
    """Validate, save under a random name, and record in the database."""
    checked = validate_upload(filename, data, kind, max_bytes)
    for existing in db.list_documents(claim_id):
        if existing.sha256 == checked.sha256 and existing.kind == kind:
            raise UploadError(f"'{checked.safe_name}' is already attached to this claim as {kind.label.lower()}.")

    folder = Path(upload_dir) / f"claim_{claim_id}"
    folder.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{checked.extension}"
    destination = folder / stored_name
    destination.write_bytes(data)
    # Read PDF text once, here, so agents never have to open an untrusted file themselves.
    text = extract_pdf_text(data) if checked.extension == ".pdf" else None
    try:
        record = db.add_document(claim_id, kind, checked.safe_name, f"claim_{claim_id}/{stored_name}",
                                 checked.mime, checked.size_bytes, checked.sha256, text)
    except Exception:
        destination.unlink(missing_ok=True)  # do not leave an orphan file behind
        raise
    log.info("Stored %s for claim %s (%d bytes)", kind.value, claim_id, checked.size_bytes)
    return record


# ---------------------------------------------------------------- instruction-like text

_INSTRUCTION_PATTERNS = [
    r"ignore (?:all |any |the )*(?:previous|prior|above|earlier) (?:instructions?|prompts?|rules?)",
    r"disregard (?:all |any |the )*(?:previous|prior|above|earlier|system)",
    r"(?:reveal|show|print) (?:the |your )?system prompt",
    r"you are now (?:an?|the) ",
    r"override (?:the )?(?:rules|policy|instructions|checks)",
    r"(?:approve|accept|pay) (?:this|the) claim (?:immediately|automatically|without)",
    r"do not (?:flag|report|mention|tell)",
]
_INSTRUCTION_REGEX = re.compile("|".join(_INSTRUCTION_PATTERNS), re.IGNORECASE)


def find_instruction_like_text(text: str) -> list[str]:
    """Return the phrases in `text` that read like orders to an AI model."""
    return [m.group(0).strip() for m in _INSTRUCTION_REGEX.finditer(text or "")]
