"""
database/init_db.py
====================
Database initialisation, schema creation, and seed data.

Run once on first startup to create all tables and insert
default data (demo user, model registry entries, etc.)

Usage:
  python database/init_db.py
  # or called automatically by api/main.py on startup
"""

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import text


async def init_db():
    """Create all tables if they don't exist."""
    from database.session import engine
    from database.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.success("All database tables created")


async def seed_demo_data():
    """Insert demo/test data for development."""
    from database.session import AsyncSessionLocal
    from database.models import User, MLModelVersion

    async with AsyncSessionLocal() as db:
        # Check if already seeded
        result = await db.execute(text("SELECT COUNT(*) FROM users"))
        count = result.scalar()
        if count and count > 0:
            logger.info("Database already seeded — skipping")
            return

        # Demo user
        demo_user = User(
            id=str(uuid.uuid4()),
            pseudonym_id="demo_0001",
            role="patient",
        )
        db.add(demo_user)

        # Clinician user
        clinician = User(
            id=str(uuid.uuid4()),
            pseudonym_id="clin_0001",
            role="clinician",
        )
        db.add(clinician)

        # Model registry entries
        models = [
            MLModelVersion(
                id=str(uuid.uuid4()),
                model_name="cnn_motor_imagery",
                version="1.0.0",
                file_path="ml/saved_models/cnn_motor_imagery.h5",
                accuracy=0.91,
                inference_ms=8.0,
                training_samples=288 * 9,   # BCI Competition IV: 9 subjects
                dataset="BCI Competition IV Dataset 2a",
                deployed=True,
                checksum_sha256=None,
            ),
            MLModelVersion(
                id=str(uuid.uuid4()),
                model_name="eegnet_motor_imagery",
                version="1.0.0",
                file_path="ml/saved_models/eegnet.pt",
                accuracy=0.89,
                inference_ms=6.0,
                training_samples=288 * 9,
                dataset="BCI Competition IV Dataset 2a",
                deployed=False,
            ),
            MLModelVersion(
                id=str(uuid.uuid4()),
                model_name="lda_baseline",
                version="1.0.0",
                file_path="ml/saved_models/lda_baseline.pkl",
                accuracy=0.75,
                inference_ms=0.5,
                training_samples=288 * 9,
                dataset="BCI Competition IV Dataset 2a",
                deployed=False,
            ),
        ]
        for m in models:
            db.add(m)

        await db.commit()
        logger.success(
            f"Seed data inserted: "
            f"2 users, {len(models)} model registry entries"
        )


async def drop_all():
    """Drop all tables — use only in development."""
    from database.session import engine
    from database.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables DROPPED")


async def main():
    """Run full database setup."""
    logger.info("Initialising BCI Platform database...")
    await init_db()
    await seed_demo_data()
    logger.success("Database setup complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BCI Platform DB init")
    parser.add_argument("--drop", action="store_true", help="Drop all tables first")
    parser.add_argument("--seed-only", action="store_true", help="Seed only (no schema)")
    args = parser.parse_args()

    async def run():
        if args.drop:
            logger.warning("Dropping all tables...")
            await drop_all()
        if not args.seed_only:
            await init_db()
        await seed_demo_data()

    asyncio.run(run())