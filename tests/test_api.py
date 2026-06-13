"""
tests/test_api.py
==================
FastAPI endpoint integration tests using httpx async client.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    """Create test FastAPI app."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.main import app as fastapi_app
    return fastapi_app


@pytest.mark.asyncio
async def test_health_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "BCI Platform" in data["name"]
    assert "IEEE 2857" in data["standards"]


@pytest.mark.asyncio
async def test_start_session(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/sessions/start", json={
            "user_id": "test_user",
            "paradigm": "motor_imagery",
            "simulate": True,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "test_user"
    assert data["paradigm"] == "motor_imagery"
    assert "session_id" in data


@pytest.mark.asyncio
async def test_eeg_quality_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/eeg/quality")
    assert resp.status_code == 200
    data = resp.json()
    assert "snr_db" in data
    assert data["snr_db"] > 35   # ISO 14155 Phase 1 target


@pytest.mark.asyncio
async def test_consent_grant(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/consent/patient_001/grant", json={
            "purposes": ["neural_processing", "audit_logging"],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "patient_001"
    assert "neural_processing" in data["consents_granted"]


@pytest.mark.asyncio
async def test_ml_status_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/ml/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["accuracy"] >= 0.90


@pytest.mark.asyncio
async def test_demo_prediction_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/ml/predict/demo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["class_name"] in ["left", "right", "feet", "rest"]
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_gradcam_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/ml/gradcam/patient_001")
    assert resp.status_code == 200
    data = resp.json()
    assert "heatmap" in data
    assert "channel_importance" in data
    assert data["ieee_2857_section"] == "§7.1 Model Transparency"


@pytest.mark.asyncio
async def test_audit_log_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/audit/")
    assert resp.status_code == 200
    data = resp.json()
    assert "entries" in data


@pytest.mark.asyncio
async def test_safe_state_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/devices/safe-state",
                                  params={"reason": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["safe_state_active"] is True