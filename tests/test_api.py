"""Unit and integration tests for FastAPI REST endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.routes import app


def test_api_health():
    """Test /health endpoint returns 200 OK and valid health payload."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["storage_ready"] is True


def test_api_path_traversal_protection():
    """Test /ingest endpoint protects against directory traversal attempts."""
    client = TestClient(app)
    resp = client.post("/ingest", json={"pdf_path": "../../../etc/shadow"})
    assert resp.status_code == 400


def test_api_exports():
    """Test /export/json and /export/csv return valid downloadable files."""
    client = TestClient(app)

    # JSON export
    res_json = client.get("/export/json")
    assert res_json.status_code == 200
    assert res_json.headers["content-type"] == "application/json"

    # CSV export
    res_csv = client.get("/export/csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
