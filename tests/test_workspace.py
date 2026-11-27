"""Tests for multi-document isolation, guest permissions, comparison, chat context, and security."""

from __future__ import annotations

from pathlib import Path

import pytest

from autospec.config import DocumentValidationError, settings
from autospec.database import DatabaseManager
from autospec.embeddings import DeterministicMockEmbedder
from autospec.schemas import ComparisonRequest, QueryRequest
from autospec.workspace_service import WorkspaceService


@pytest.fixture
def isolated_workspace(temp_dir: Path, monkeypatch) -> WorkspaceService:
    monkeypatch.setattr(settings, "storage_dir", temp_dir)
    monkeypatch.setattr(settings, "index_dir", temp_dir / "indexes")
    monkeypatch.setattr(settings, "cache_dir", temp_dir / "cache")
    monkeypatch.setattr(settings, "demo_dir", temp_dir / "demo")
    db = DatabaseManager(database_url=f"sqlite:///{(temp_dir / 'test.db').as_posix()}")
    svc = WorkspaceService(db=db, embedder=DeterministicMockEmbedder(384))
    svc.initialize_demo_manuals()
    yield svc
    svc.shutdown()


def test_demo_manuals_seeded_and_readonly(isolated_workspace: WorkspaceService):
    docs = isolated_workspace.list_documents("session_alpha")
    assert len(docs) == 2
    assert all(d.is_demo and d.read_only for d in docs)

    # Attempting to delete or rename a read-only demo manual must fail with read_only
    err = isolated_workspace.delete_document(docs[0].document_id, "session_alpha")
    assert err == "read_only"
    updated, r_err = isolated_workspace.db.update_document_metadata(
        docs[0].document_id, "session_alpha", display_name="Hacked"
    )
    assert updated is None and r_err == "read_only"


def test_multi_user_session_isolation(isolated_workspace: WorkspaceService, sample_synthetic_pdf: Path):
    pdf_bytes = sample_synthetic_pdf.read_bytes()

    resp = isolated_workspace.ingest_upload_sync(
        data=pdf_bytes,
        raw_filename="user_a_manual.pdf",
        session_id="user_a",
        manufacturer="Ford",
        model="F-150",
        year=2014,
    )
    assert resp.status == "success"

    # User A sees demo + own upload (3); User B sees ONLY demo (2)
    docs_a = isolated_workspace.list_documents("user_a")
    docs_b = isolated_workspace.list_documents("user_b")
    assert len(docs_a) == 3
    assert len(docs_b) == 2
    assert all(d.document_id != resp.document_id for d in docs_b)

    # User B cannot access or delete User A's document
    assert isolated_workspace.get_document(resp.document_id, "user_b") is None
    assert isolated_workspace.delete_document(resp.document_id, "user_b") == "not_found"

    # Search history is strictly isolated
    isolated_workspace.query(QueryRequest(query="Brake caliper anchor plate bolts"), session_id="user_a")
    assert len(isolated_workspace.get_history("user_a")) == 1
    assert len(isolated_workspace.get_history("user_b")) == 0


def test_malicious_upload_rejected(isolated_workspace: WorkspaceService):
    with pytest.raises(DocumentValidationError, match="valid PDF header"):
        isolated_workspace.ingest_upload_sync(
            data=b"<html><script>alert(1)</script></html>",
            raw_filename="evil.pdf",
            session_id="user_a",
        )


def test_vehicle_comparison_engine(isolated_workspace: WorkspaceService):
    docs = isolated_workspace.list_documents("public")
    req = ComparisonRequest(
        document_ids=[d.document_id for d in docs],
        subsystem="front_brakes",
        preferred_torque_unit="lb-ft",
    )
    report = isolated_workspace.compare_vehicles(req, "public")
    assert len(report.vehicles) == 2
    assert report.total_rows > 0

    guide_row = next(
        (r for r in report.rows if "guide pin" in r.canonical_component.lower()),
        None,
    )
    assert guide_row is not None
    assert guide_row.comparable is True
    assert guide_row.has_variance is True
    # Verify original source units preserved alongside lb-ft normalized values
    assert all(c.original_unit == "Nm" and c.normalized_unit == "lb-ft" for c in guide_row.cells)


def test_conversational_follow_up_context(isolated_workspace: WorkspaceService):
    conv_id = "thread_1"
    r1 = isolated_workspace.query(
        QueryRequest(
            query="What is the cylinder head bolt torque on 2022 Apex M400?",
            conversation_id=conv_id,
            retrieval_mode="bm25",
        ),
        session_id="conv_user",
    )
    assert r1.total_results >= 1

    # Follow-up question relying on conversational context
    r2 = isolated_workspace.query(
        QueryRequest(
            query="What about the tightening sequence?",
            conversation_id=conv_id,
            retrieval_mode="bm25",
        ),
        session_id="conv_user",
    )
    assert r2.status == "answered"
    assert r2.total_results >= 1
    assert "sequence" in r2.results[0].source_quote.lower() or "tighten" in r2.results[0].source_quote.lower()


def test_concurrent_background_job_upload(isolated_workspace: WorkspaceService, sample_synthetic_pdf: Path):
    import time

    job1 = isolated_workspace.submit_upload_job(
        data=sample_synthetic_pdf.read_bytes(),
        raw_filename="concurrent_1.pdf",
        session_id="conc_user",
    )
    job2 = isolated_workspace.submit_upload_job(
        data=sample_synthetic_pdf.read_bytes(),
        raw_filename="concurrent_2.pdf",
        session_id="conc_user",
    )

    deadline = time.time() + 15
    while time.time() < deadline:
        s1 = isolated_workspace.db.get_job(job1.job_id, "conc_user")
        s2 = isolated_workspace.db.get_job(job2.job_id, "conc_user")
        if s1 and s2 and s1.status == "completed" and s2.status == "completed":
            break
        time.sleep(0.1)

    s1 = isolated_workspace.db.get_job(job1.job_id, "conc_user")
    s2 = isolated_workspace.db.get_job(job2.job_id, "conc_user")
    assert s1 is not None and s1.status == "completed"
    assert s2 is not None and s2.status == "completed"


def test_duplicate_upload_links_existing_document(
    isolated_workspace: WorkspaceService, sample_synthetic_pdf: Path
):
    raw = sample_synthetic_pdf.read_bytes()
    first = isolated_workspace.ingest_upload_sync(
        data=raw, raw_filename="first.pdf", session_id="dup_user"
    )
    assert first.status == "success"

    dup_job = isolated_workspace.submit_upload_job(
        data=raw, raw_filename="duplicate.pdf", session_id="dup_user"
    )
    assert dup_job.status == "completed"
    assert dup_job.is_duplicate is True
    assert dup_job.document_id == first.document_id


def test_db_persistence_recovers_after_local_cache_loss(
    isolated_workspace: WorkspaceService, sample_synthetic_pdf: Path
):
    import shutil

    raw = sample_synthetic_pdf.read_bytes()
    ingested = isolated_workspace.ingest_upload_sync(
        data=raw,
        raw_filename="ephemeral_test.pdf",
        session_id="restart_user",
        manufacturer="Ford",
        model="F-150",
        year=2014,
    )

    # Simulate container restart / ephemeral disk wipe
    isolated_workspace._loaded_pipelines.clear()
    idx_dir = settings.get_index_dir() / "collections" / ingested.document_id
    if idx_dir.exists():
        shutil.rmtree(idx_dir, ignore_errors=True)
    rec = isolated_workspace.db.get_authorized_document(ingested.document_id, "restart_user")
    assert rec is not None and rec.pdf_path
    Path(rec.pdf_path).unlink(missing_ok=True)

    # Query and PDF download must still succeed via DB records + pdf_data BYTEA
    pdf_bytes, _, _ = isolated_workspace.get_pdf_content(ingested.document_id, "restart_user")
    assert pdf_bytes is not None and pdf_bytes.startswith(b"%PDF")

    q_resp = isolated_workspace.query(
        QueryRequest(
            query="Brake caliper anchor plate bolts",
            document_ids=[ingested.document_id],
            retrieval_mode="bm25",
        ),
        session_id="restart_user",
    )
    assert q_resp.total_results >= 1
    assert q_resp.results[0].value


