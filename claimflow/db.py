"""
SQLite storage. Four tables: claims, documents, agent_runs, reviews.

A new connection is opened for each operation. SQLite connections are cheap, and
Streamlit runs scripts on several threads, so sharing one connection would be
the harder thing to get right. Every public method is one transaction.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

from .errors import DatabaseError, InvalidInputError, WorkflowError
from .logging_setup import get_logger
from .models import (
    ALLOWED_TRANSITIONS, DECISION_TO_STATUS, AgentResult, ClaimRecord, ClaimStatus,
    DocumentKind, DocumentRecord, IncidentType, NewClaim, ReviewDecision, ReviewRecord,
    SharedFile, can_transition, is_override, utc_now,
)

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_number  TEXT UNIQUE,
    status        TEXT NOT NULL,
    claimant_name TEXT NOT NULL,
    policy_number TEXT NOT NULL,
    incident_date TEXT NOT NULL,
    incident_type TEXT NOT NULL,
    description   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id       INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,
    original_name  TEXT NOT NULL,
    stored_path    TEXT NOT NULL,
    mime           TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    sha256         TEXT NOT NULL,
    extracted_text TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id    INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    step        TEXT NOT NULL,
    status      TEXT NOT NULL,
    result_json TEXT NOT NULL,
    error       TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    model       TEXT,
    cached      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id   INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    decision   TEXT NOT NULL,
    reviewer   TEXT NOT NULL,
    notes      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    ai_recommendation TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_claim ON documents(claim_id);
CREATE INDEX IF NOT EXISTS idx_runs_claim_step ON agent_runs(claim_id, step);
CREATE INDEX IF NOT EXISTS idx_reviews_claim ON reviews(claim_id);
"""


def _claim(row: sqlite3.Row) -> ClaimRecord:
    return ClaimRecord(
        id=row["id"], claim_number=row["claim_number"], status=ClaimStatus(row["status"]),
        claimant_name=row["claimant_name"], policy_number=row["policy_number"],
        incident_date=date.fromisoformat(row["incident_date"]),
        incident_type=IncidentType(row["incident_type"]), description=row["description"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"], claim_id=row["claim_id"], kind=DocumentKind(row["kind"]),
        original_name=row["original_name"], stored_path=row["stored_path"], mime=row["mime"],
        size_bytes=row["size_bytes"], sha256=row["sha256"], extracted_text=row["extracted_text"],
        created_at=row["created_at"],
    )


def _review(row: sqlite3.Row) -> ReviewRecord:
    return ReviewRecord(
        id=row["id"], claim_id=row["claim_id"], decision=ReviewDecision(row["decision"]),
        reviewer=row["reviewer"], notes=row["notes"], created_at=row["created_at"],
        ai_recommendation=row["ai_recommendation"] if "ai_recommendation" in row.keys() else None,
    )


def _result(row: sqlite3.Row) -> AgentResult:
    result = AgentResult.from_json(row["result_json"])
    result.created_at = row["created_at"]
    return result


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            if conn is not None:
                conn.rollback()
            log.error("SQLite error: %s", exc)
            raise DatabaseError("The database could not complete the request.",
                                hint="Check that the data folder is writable.") from exc
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                conn.close()

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # Databases made by earlier builds lack this column. CREATE TABLE IF NOT EXISTS leaves them alone.
            if "ai_recommendation" not in {r["name"] for r in conn.execute("PRAGMA table_info(reviews)")}:
                conn.execute("ALTER TABLE reviews ADD COLUMN ai_recommendation TEXT")
        try:
            with self.connect() as conn:
                conn.execute("PRAGMA journal_mode = WAL")
        except DatabaseError:
            pass  # WAL is a speed-up, not a requirement
        log.info("Database ready at %s", self.path)

    # ------------------------------------------------------------ claims

    def create_claim(self, new: NewClaim) -> ClaimRecord:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO claims (status, claimant_name, policy_number, incident_date, "
                "incident_type, description, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (ClaimStatus.SUBMITTED.value, new.claimant_name, new.policy_number,
                 new.incident_date.isoformat(), new.incident_type.value, new.description, now, now),
            )
            claim_id = cursor.lastrowid
            number = f"CF-{now[:4]}-{claim_id:04d}"
            conn.execute("UPDATE claims SET claim_number = ? WHERE id = ?", (number, claim_id))
            row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        log.info("Created claim %s", number)
        return _claim(row)

    def get_claim(self, claim_id: int) -> ClaimRecord:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        if row is None:
            raise DatabaseError(f"Claim {claim_id} was not found.")
        return _claim(row)

    def list_claims(self, limit: int = 200) -> list[ClaimRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM claims ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_claim(r) for r in rows]

    def _set_status(self, conn: sqlite3.Connection, claim_id: int, new: ClaimStatus) -> None:
        row = conn.execute("SELECT status FROM claims WHERE id = ?", (claim_id,)).fetchone()
        if row is None:
            raise DatabaseError(f"Claim {claim_id} was not found.")
        current = ClaimStatus(row["status"])
        if current != new and not can_transition(current, new):
            raise WorkflowError(f"A claim that is {current.label.lower()} cannot move to {new.label.lower()}.")
        conn.execute("UPDATE claims SET status = ?, updated_at = ? WHERE id = ?",
                     (new.value, utc_now(), claim_id))

    def update_claim_status(self, claim_id: int, new: ClaimStatus) -> ClaimRecord:
        with self.connect() as conn:
            self._set_status(conn, claim_id, new)
        return self.get_claim(claim_id)

    # ------------------------------------------------------------ documents

    def add_document(self, claim_id: int, kind: DocumentKind, original_name: str, stored_path: str,
                     mime: str, size_bytes: int, sha256: str,
                     extracted_text: str | None = None) -> DocumentRecord:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO documents (claim_id, kind, original_name, stored_path, mime, size_bytes, "
                "sha256, extracted_text, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (claim_id, kind.value, original_name, stored_path, mime, size_bytes, sha256,
                 extracted_text, utc_now()),
            )
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (cursor.lastrowid,)).fetchone()
            conn.execute("UPDATE claims SET updated_at = ? WHERE id = ?", (utc_now(), claim_id))
        return _document(row)

    def list_documents(self, claim_id: int) -> list[DocumentRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM documents WHERE claim_id = ? ORDER BY id", (claim_id,)).fetchall()
        return [_document(r) for r in rows]

    def shared_documents(self, claim_id: int) -> list[SharedFile]:
        """Files on this claim that are identical (same SHA-256) to files on other claims."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT d.original_name, d.kind, c.claim_number FROM documents d "
                "JOIN documents o ON o.sha256 = d.sha256 AND o.claim_id != d.claim_id "
                "JOIN claims c ON c.id = o.claim_id WHERE d.claim_id = ? ORDER BY d.id, c.id", (claim_id,),
            ).fetchall()
        grouped: dict[tuple[str, str], list[str]] = {}
        for row in rows:
            grouped.setdefault((row["original_name"], row["kind"]), []).append(row["claim_number"])
        return [SharedFile(name, DocumentKind(kind), tuple(numbers)) for (name, kind), numbers in grouped.items()]

    def set_extracted_text(self, document_id: int, text: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE documents SET extracted_text = ? WHERE id = ?", (text, document_id))

    # ------------------------------------------------------------ agent runs

    def save_agent_result(self, claim_id: int, result: AgentResult) -> None:
        """Every run is kept. The latest per step is what the UI shows; the rest is the trace."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO agent_runs (claim_id, step, status, result_json, error, duration_ms, model, "
                "cached, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (claim_id, result.step, result.status.value, result.to_json(), result.error,
                 result.duration_ms, result.model, int(result.cached), utc_now()),
            )

    def latest_agent_results(self, claim_id: int) -> dict[str, AgentResult]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_runs WHERE id IN "
                "(SELECT MAX(id) FROM agent_runs WHERE claim_id = ? GROUP BY step)", (claim_id,),
            ).fetchall()
        return {r["step"]: _result(r) for r in rows}

    def agent_history(self, claim_id: int) -> list[AgentResult]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM agent_runs WHERE claim_id = ? ORDER BY id", (claim_id,)).fetchall()
        return [_result(r) for r in rows]

    # ------------------------------------------------------------ reviews

    def add_review(self, claim_id: int, decision: ReviewDecision, reviewer: str, notes: str = "",
                   ai_recommendation: str | None = None) -> ReviewRecord:
        """
        Record a human decision and move the claim, in one transaction. Only a claim that is
        waiting for review can be decided. A decision that differs from the AI recommendation
        needs a written reason, and the recommendation is stored with the decision for the audit trail.
        """
        reviewer = (reviewer or "").strip()
        if len(reviewer) < 2:
            raise InvalidInputError("Enter your name before recording a decision.")
        if is_override(decision, ai_recommendation) and len((notes or "").strip()) < 5:
            raise InvalidInputError("This decision differs from the AI recommendation. Add a note explaining why.")
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM claims WHERE id = ?", (claim_id,)).fetchone()
            if row is None:
                raise DatabaseError(f"Claim {claim_id} was not found.")
            if ClaimStatus(row["status"]) != ClaimStatus.AWAITING_REVIEW:
                raise WorkflowError("This claim is not waiting for review.",
                                    hint="Run the agents first, or the claim already has a final decision.")
            cursor = conn.execute(
                "INSERT INTO reviews (claim_id, decision, reviewer, notes, created_at, ai_recommendation) "
                "VALUES (?,?,?,?,?,?)",
                (claim_id, decision.value, reviewer, (notes or "").strip(), utc_now(), ai_recommendation),
            )
            self._set_status(conn, claim_id, DECISION_TO_STATUS[decision])
            saved = conn.execute("SELECT * FROM reviews WHERE id = ?", (cursor.lastrowid,)).fetchone()
        log.info("Claim %s: %s by %s", claim_id, decision.value, reviewer)
        return _review(saved)

    def list_reviews(self, claim_id: int) -> list[ReviewRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM reviews WHERE claim_id = ? ORDER BY id DESC", (claim_id,)).fetchall()
        return [_review(r) for r in rows]

    # ------------------------------------------------------------ dashboard

    def dashboard_stats(self) -> dict:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            counts = {s.value: 0 for s in ClaimStatus}
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM claims GROUP BY status"):
                counts[row["status"]] = row["n"]
            latest = "SELECT MAX(id) FROM agent_runs GROUP BY claim_id, step"
            with_warnings = conn.execute(
                f"SELECT COUNT(DISTINCT claim_id) FROM agent_runs WHERE id IN ({latest}) "
                "AND status IN ('warning', 'failed')").fetchone()[0]
            failed = conn.execute(
                f"SELECT COUNT(*) FROM agent_runs WHERE id IN ({latest}) AND status = 'failed'").fetchone()[0]
        return {
            "total": total,
            "by_status": counts,
            "awaiting_review": counts[ClaimStatus.AWAITING_REVIEW.value],
            "claims_with_warnings": with_warnings,
            "failed_steps": failed,
        }
