"""
database/db.py

Session management and all query/write functions.
Everything that touches the DB goes through here.

Public API:
    init_db()                   create tables if not exist
    save_prediction(result)     write one PredictionResult
    save_predictions(results)   write a list (batch) in one transaction
    get_history(limit, ...)     fetch recent predictions
    get_batch(batch_id)         fetch all rows for one batch run
    get_stats()                 summary counts for the UI dashboard
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, Session


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_URL, DB_PATH
from database.schema import Base, Prediction

log = logging.getLogger(__name__)

# ── Engine singleton ──────────────────────────────────────────────────────
_engine  = None
_Session = None


def _init_engine():
    global _engine, _Session
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine  = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},  # required for SQLite + Streamlit
            echo=False,
        )
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)


@contextmanager
def _session() -> Session:
    """Internal context manager — commits or rolls back cleanly."""
    _init_engine()
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_ts(value) -> datetime:
    """Normalise timestamp — accepts ISO string, datetime, or None."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _build_record(data: dict) -> Prediction:
    """Map a PredictionResult dict to a Prediction ORM row."""
    return Prediction(
        image_url         = data["image_url"],
        run_type          = data["run_type"],
        run_id            = data["run_id"],
        predicted_gender  = data["predicted_gender"],
        predicted_sleeve  = data["predicted_sleeve"],
        confidence_gender = data.get("confidence_gender"),
        confidence_sleeve = data.get("confidence_sleeve"),
        model_name        = data.get("model_name", ""),
        model_version     = data.get("model_version", ""),
        timestamp         = _parse_ts(data.get("timestamp")),
        status            = data.get("status", "success"),
        error_message     = data.get("error_message", ""),
    )


def _to_dict(result) -> dict:
    """Accept PredictionResult dataclass or plain dict."""
    return result.as_dict() if hasattr(result, "as_dict") else result


# ── Setup ─────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist. Idempotent."""
    _init_engine()
    Base.metadata.create_all(_engine)
    log.info(f"Database ready at {DB_PATH}")


# ── Write ─────────────────────────────────────────────────────────────────

def save_prediction(result) -> Prediction:
    record = _build_record(_to_dict(result))
    with _session() as s:
        s.add(record)
        s.flush()
    return record


def save_predictions(results: list) -> list[Prediction]:
    if not results:
        return []
    records = [_build_record(_to_dict(r)) for r in results]
    with _session() as s:
        s.add_all(records)
    log.info(f"Saved {len(records)} predictions")
    return records


# ── Read ──────────────────────────────────────────────────────────────────

def get_history(
    limit:    int           = 100,
    run_type: Optional[str] = None,
    status:   Optional[str] = None,
) -> list[dict]:
    """Fetch recent predictions newest-first. Returns dicts safe for Streamlit."""
    with _session() as s:
        q = s.query(Prediction).order_by(desc(Prediction.timestamp))
        if run_type:
            q = q.filter(Prediction.run_type == run_type)
        if status:
            q = q.filter(Prediction.status == status)
        return [r.as_dict() for r in q.limit(limit).all()]


def get_batch(batch_id: str) -> list[dict]:
    """Fetch all rows for one batch run, in insertion order."""
    with _session() as s:
        rows = (
            s.query(Prediction)
            .filter(Prediction.run_id == batch_id)
            .order_by(Prediction.id)
            .all()
        )
        return [r.as_dict() for r in rows]


def get_stats() -> dict:
    """
    Summary counts for the History page dashboard.
    Runs two queries — one for totals, one for per-class breakdowns.
    """
    with _session() as s:
        # ── query 1: scalar counts ────────────────────────────────────────
        total        = s.query(func.count(Prediction.id)).scalar() or 0
        total_single = s.query(func.count(Prediction.id)).filter(
                           Prediction.run_type == "single").scalar() or 0
        total_batch  = s.query(func.count(Prediction.id)).filter(
                           Prediction.run_type == "batch").scalar() or 0
        total_errors = s.query(func.count(Prediction.id)).filter(
                           Prediction.status == "error").scalar() or 0

        # ── query 2: per-class breakdowns (success rows only) ─────────────
        gender_counts = dict(
            s.query(Prediction.predicted_gender, func.count(Prediction.id))
            .filter(Prediction.status == "success")
            .group_by(Prediction.predicted_gender)
            .all()
        )
        sleeve_counts = dict(
            s.query(Prediction.predicted_sleeve, func.count(Prediction.id))
            .filter(Prediction.status == "success")
            .group_by(Prediction.predicted_sleeve)
            .all()
        )

    return {
        "total":         total,
        "total_single":  total_single,
        "total_batch":   total_batch,
        "total_errors":  total_errors,
        "gender_counts": gender_counts,
        "sleeve_counts": sleeve_counts,
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    init_db()
    print(get_stats())