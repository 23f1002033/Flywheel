import datetime
from sqlalchemy import (Boolean, DateTime, Float, Integer, LargeBinary,
                        String, Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from app.config import get_settings


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, index=True)
    route: Mapped[str] = mapped_column(String(16), index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    model: Mapped[str] = mapped_column(String(128), default="")
    model_version: Mapped[str] = mapped_column(String(32), default="")
    prompt: Mapped[str] = mapped_column(Text, default="")
    response: Mapped[str] = mapped_column(Text, default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    counterfactual_usd: Mapped[float] = mapped_column(Float, default=0.0)
    saved_usd: Mapped[float] = mapped_column(Float, default=0.0)
    co2_saved_grams: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


_engine = create_engine(get_settings().db_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)