"""
database/schema.py

SQLAlchemy ORM definition for the predictions table.
Kept separate from db.py so the table structure is easy to read at a glance.

Table: predictions
    One row per image prediction.
    Single run  → run_type="single", run_id=UUID unique per call
    Batch run   → run_type="batch",  run_id=shared UUID for the whole batch
"""

from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, DateTime, Integer, Text, Index
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # image reference
    image_url = Column(Text,        nullable=False)

    # run metadata
    run_type  = Column(String(10),  nullable=False)   # "single" | "batch"
    run_id    = Column(String(36),  nullable=False)   # UUID

    # predictions
    predicted_gender  = Column(String(10), nullable=False)
    predicted_sleeve  = Column(String(20), nullable=False)
    confidence_gender = Column(Float,      nullable=True)
    confidence_sleeve = Column(Float,      nullable=True)

    # model metadata
    model_name    = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)

    # bookkeeping
    timestamp     = Column(DateTime(timezone=True), nullable=False,
                           default=lambda: datetime.now(timezone.utc))
    status        = Column(String(10), nullable=False, default="success")
    error_message = Column(Text,       nullable=True,  default="")

    # indexes — only on high-cardinality / frequently sorted columns
    __table_args__ = (
        Index("ix_run_id",    "run_id"),     # get_batch: WHERE run_id = ?
        Index("ix_timestamp", "timestamp"),  # get_history: ORDER BY timestamp DESC
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} run_type={self.run_type} "
            f"gender={self.predicted_gender} sleeve={self.predicted_sleeve} "
            f"status={self.status}>"
        )

    def as_dict(self) -> dict:
        return {
            "id":               self.id,
            "image_url":        self.image_url,
            "run_type":         self.run_type,
            "run_id":           self.run_id,
            "predicted_gender": self.predicted_gender,
            "predicted_sleeve": self.predicted_sleeve,
            "confidence_gender": round(self.confidence_gender or 0.0, 4),
            "confidence_sleeve": round(self.confidence_sleeve or 0.0, 4),
            "model_name":       self.model_name,
            "model_version":    self.model_version,
            "timestamp":        self.timestamp.isoformat() if self.timestamp else "",
            "status":           self.status,
            "error_message":    self.error_message or "",
        }