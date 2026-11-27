"""Unit and integration tests for FastAPI v1/legacy REST endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.routes import app


def test_api_health():
    """Test /health returns immediately (non-blocking) and background demo finishes cleanly."""
    import time

    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["storage_ready"] is True
        assert data["database_ready"] is True
        assert data["database_backend"] in ("sqlite", "postgresql")
        assert data["demo_state"] in ("initializing", "ready")
        assert "X-Request-ID" in resp.headers
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

        # Wait for non-blocking background demo bootstrap to reach ready
        deadline = time.time() + 30
        while data.get("demo_state") != "ready" and time.time() < deadline:
            time.sleep(0.1)
            data = client.get("/api/v1/health").json()
        assert data["demo_state"] == "ready"
        assert data["demo_ready"] is True


def test_api_rejects_local_path_json_and_non_pdf_upload():
    """Test /ingest rejects JSON pdf_path payloads and non-PDF multipart files."""
    with TestClient(app) as client:
        resp_json = client.post("/ingest", json={"pdf_path": "../../../etc/shadow"})
        assert resp_json.status_code == 400

        resp_evil = client.post(
            "/api/v1/documents/upload",
            files={"file": ("../../etc/passwd.pdf", b"<html>bad</html>", "application/pdf")},
        )
        assert resp_evil.status_code == 400


def test_api_overview_demo_and_evaluation_endpoints():
    """Test /overview, /demo, and /evaluation/runs return valid structured payloads."""
    with TestClient(app) as client:
        ov = client.get("/api/v1/overview")
        assert ov.status_code == 200
        ov_data = ov.json()
        assert ov_data["demo_document_count"] >= 2
        assert ov_data["total_specifications"] > 0

        demo = client.get("/api/v1/demo")
        assert demo.status_code == 200
        assert demo.json()["demo_ready"] is True
        assert len(demo.json()["sample_queries"]) >= 3

        runs = client.get("/api/v1/evaluation/runs")
        assert runs.status_code == 200
        assert isinstance(runs.json(), list)


def test_api_documents_and_query():
    """Test /api/v1/documents, /pdf binary stream, and /api/v1/query on seeded demo manuals."""
    with TestClient(app) as client:
        docs_resp = client.get("/api/v1/documents")
        assert docs_resp.status_code == 200
        docs = docs_resp.json()
        assert len(docs) >= 2

        pdf_resp = client.get(f"/api/v1/documents/{docs[0]['document_id']}/pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        assert pdf_resp.content.startswith(b"%PDF")

        q_resp = client.post(
            "/api/v1/query",
            json={
                "query": "What is the torque for front brake caliper guide pin bolts on 2022 Apex M400?",
                "retrieval_mode": "bm25",
                "top_k": 3,
            },
        )
        assert q_resp.status_code == 200
        payload = q_resp.json()
        assert payload["total_results"] >= 1
        assert payload["results"][0]["value"] == "37"


def test_api_exports():
    """Test /export/json and /export/csv return valid downloadable files."""
    with TestClient(app) as client:
        res_json = client.get("/api/v1/export/json")
        assert res_json.status_code == 200
        assert res_json.headers["content-type"] == "application/json"

        res_csv = client.get("/api/v1/export/csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]
