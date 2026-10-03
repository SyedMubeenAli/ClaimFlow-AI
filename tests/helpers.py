"""Shared test helpers: throwaway folders, a ready container, and a sample claim."""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from claimflow.config import ROOT
from claimflow.container import AppContainer, build_container
from claimflow.db import Database
from claimflow.logging_setup import setup_logging
from claimflow.models import ClaimRecord, NewClaim


@contextmanager
def temp_container(**env_overrides: str):
    """A full container whose database, uploads, cache and logs live in a temp folder."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = {
            "DATABASE_PATH": f"{tmp}/test.db", "UPLOAD_DIR": f"{tmp}/uploads",
            "LLM_CACHE_DIR": f"{tmp}/cache", "LOG_DIR": f"{tmp}/logs",
            "LLM_RETRY_DELAY_S": "0", "LOG_LEVEL": "ERROR",
            "EMBEDDING_BACKEND": "hashing", "INDEX_DIR": f"{tmp}/index", "DEMO_MODE": "false",
        }
        env.update(env_overrides)
        try:
            yield build_container(env), Path(tmp)
        finally:
            setup_logging("ERROR")  # closes the log file so the folder can be deleted


@contextmanager
def temp_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = Database(Path(tmp) / "test.db")
        db.init()
        yield db, Path(tmp)


def sample_claim(**overrides) -> NewClaim:
    # A recent date keeps tests independent of the calendar. The late-entry indicator looks at the claim's age.
    values = dict(claimant_name="Ayesha Khan", policy_number="POL-1001", incident_date=date.today() - timedelta(days=2),
                  incident_type="collision", description="Rear-ended at a traffic light on Shahrah-e-Faisal.")
    values.update(overrides)
    return NewClaim.from_form(**values)


def make_claim(db: Database, **overrides) -> ClaimRecord:
    return db.create_claim(sample_claim(**overrides))


@contextmanager
def quiet_logs():
    """Hide log output for a test that triggers an expected error and its logged traceback."""
    logger = logging.getLogger("claimflow")
    old_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def make_pdf(pages: list[list[str]]) -> bytes:
    """A small valid PDF with one list of text lines per page. Enough for pypdf to read, no extra library."""
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    count = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, lines in enumerate(pages):
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                       "/Resources << /Font << /F1 3 0 R >> >> >>".encode())
        stream = ("BT /F1 11 Tf 50 740 Td 14 TL " + " ".join(f"({esc(line)}) Tj T*" for line in lines) + " ET").encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def attach(container, claim, kind, filename: str, pages: list[list[str]]):
    """Store a generated PDF on a claim through the real upload path."""
    from claimflow.uploads import store_document
    return store_document(container.db, container.settings.upload_dir, claim.id, kind, filename,
                          make_pdf(pages), container.settings.max_upload_bytes)


POLICY_PAGES = [
    ["Motor Insurance Policy POL-1001",
     "Section 3. Collision cover.",
     "We pay for accidental damage to the insured vehicle caused by collision with another vehicle or object.",
     "The most we will pay for any one claim is the sum insured of 500,000."],
    ["Section 4. Excess.",
     "You must pay the first 25,000 of every claim for accidental damage.",
     "Section 7. Exclusions.",
     "We do not cover damage that happens while the vehicle is used for racing or speed testing.",
     "We do not cover wear and tear or mechanical breakdown."],
]
REPORT_PAGES = [["Accident report",
                 "Date of accident: 1 September 2026.",
                 "Location: Shahrah-e-Faisal, Karachi.",
                 "Vehicle: Honda City 2019.",
                 "Police report number: KHI-4411."]]
QUOTATION_PAGES = [["Repair quotation",
                    "Front bumper replacement 40,000.",
                    "Paint and labour 10,000.",
                    "Total 50,000."]]


def png_bytes(size=(64, 48), color=(180, 40, 40)) -> bytes:
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


def attach_photo(container, claim, filename="dent.png", data: bytes | None = None):
    from claimflow.models import DocumentKind
    from claimflow.uploads import store_document
    return store_document(container.db, container.settings.upload_dir, claim.id, DocumentKind.DAMAGE_PHOTO,
                          filename, data or png_bytes(), container.settings.max_upload_bytes)


def step_result(step: str, schema, status=None, evidence=None):
    """An AgentResult as an earlier agent would have left it on the shared state."""
    from claimflow.models import AgentResult, StepStatus
    return AgentResult(step, status or StepStatus.DONE, output=schema.to_dict(), evidence=evidence or [])


def ctx_for(container, claim, results=None, llm=None, shared=None):
    """An AgentContext over a claim's real documents and the given earlier results."""
    from claimflow.agents.base import AgentContext
    from claimflow.llm import FakeProvider
    from claimflow.models import ClaimState
    state = ClaimState(claim, container.db.list_documents(claim.id), dict(results or {}), list(shared or []))
    return AgentContext(state=state, llm=llm or FakeProvider(), settings=container.settings)


class Skip(Exception):
    """Raised by a test that needs something this machine does not have. The runner reports it as skipped."""


SAMPLE_POLICY = ROOT / "samples" / "sample_motor_policy.pdf"


def attach_sample_policy(container, claim):
    """Attach the shipped demo policy through the real upload path."""
    from claimflow.models import DocumentKind
    from claimflow.uploads import store_document
    return store_document(container.db, container.settings.upload_dir, claim.id, DocumentKind.POLICY,
                          SAMPLE_POLICY.name, SAMPLE_POLICY.read_bytes(), container.settings.max_upload_bytes)
