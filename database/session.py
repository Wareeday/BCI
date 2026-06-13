"""
database/session.py
====================
Async SQLAlchemy database session management.

Supports:
  - SQLite (development, no server needed)
  - PostgreSQL (production, via DATABASE_URL env var)

Connection pool configured for clinical workload:
  100+ concurrent users per hospital deployment.
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from loguru import logger

from database.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./bci_platform.db")

# Convert sync postgres URL to async format
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,   # Set True to log all SQL in dev
    pool_pre_ping=True,
    # Pool settings for 100+ concurrent users
    pool_size=20 if "postgresql" in DATABASE_URL else 5,
    max_overflow=30 if "postgresql" in DATABASE_URL else 0,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"Database initialised: {DATABASE_URL.split('///')[0]}")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()