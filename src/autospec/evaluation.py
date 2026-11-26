"""Rigorous empirical evaluation and ablation framework for AutoSpec AI."""

from __future__ import annotations

import json
import math
import re
import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autospec.embeddings import DeterministicMockEmbedder, EmbedderProtocol, get_embedder
from autospec.hybrid_retriever import RetrievalMode
from autospec.pipeline import AutoSpecPipeline
from autospec.schemas import VerificationStatus
from autospec.units import are_units_compatible, normalize_unit_symbol, values_equivalent


@dataclass
class BenchmarkCase:
    """Source-verified benchmark test case."""

    id: str
    split: str  # 'tuning' or 'held_out'
    category: str
    query: str
    expected_document: str | None
    expected_component: str | None
    expected_value: str | None
    expected_unit: str | None
    expected_page: int | None
    expected_subsystem: str | None
    is_answerable: bool = True
    is_ambiguous_query: bool = False
    is_adversarial: bool = False


def _component_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    e_tokens = set(re.findall(r"[a-z0-9\-]{2,}", expected.lower()))
    a_tokens = set(re.findall(r"[a-z0-9\-]{2,}", actual.lower()))
    if not e_tokens:
        return False
    if expected.lower() in actual.lower() or actual.lower() in expected.lower():
        return True
    overlap = len(e_tokens & a_tokens) / len(e_tokens)
    return overlap >= 0.6


def _unit_matches(expected_unit: str | None, actual_unit: str | None, actual_alt_unit: str | None = None) -> bool:
    if not expected_unit:
        return True
    exp_norm = normalize_unit_symbol(expected_unit)
    act_norm = normalize_unit_symbol(actual_unit)
    alt_norm = normalize_unit_symbol(actual_alt_unit)
    if exp_norm == act_norm or (alt_norm and exp_norm == alt_norm):
        return True
    if exp_norm and act_norm and are_units_compatible(exp_norm, act_norm):
        return True
    return False


def _is_chunk_relevant(chunk: Any, tc: BenchmarkCase) -> bool:
    """Strict retrieval relevance check requiring source doc + page + component/value alignment."""
    if not tc.is_answerable:
        return False
    if tc.expected_document and chunk.document_name != tc.expected_document:
        return False
    if tc.expected_page and chunk.page_number != tc.expected_page:
        return False
    if tc.expected_component and chunk.component:
        if _component_matches(tc.expected_component, chunk.component):
            return True
    if tc.expected_value and (
        values_equivalent(tc.expected_value, tc.expected_unit, chunk.primary_value, chunk.primary_unit)
        or (tc.expected_value.lower() in (chunk.raw_evidence or "").lower())
    ):
        return True
    return False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(s[int(k)], 2)
    return round(s[f] * (c - k) + s[c] * (k - f), 2)


@dataclass
class EvaluationRunner:
    """Executes full retrieval, extraction, grounding, and ablation benchmarks."""

    embedder: EmbedderProtocol
    top_k: int = 5

    def evaluate_cases(
        self,
        pipelines: dict[str, AutoSpecPipeline],
        cases: list[BenchmarkCase],
        retrieval_mode: RetrievalMode = "hybrid_rerank",
        include_details: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a list of benchmark cases across indexed document pipelines."""
        answerable_cases = [c for c in cases if c.is_answerable]
        negative_cases = [c for c in cases if not c.is_answerable]

        precisions: list[float] = []
        recalls: list[float] = []
        rrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []

        comp_ok_count = 0
        val_ok_count = 0
        unit_ok_count = 0
        citation_ok_count = 0
        grounded_ok_count = 0
        answerable_grounding_failures = 0

        neg_rejected_count = 0
        neg_hallucinated_count = 0

        by_category: dict[str, dict[str, int]] = {}
        details: list[dict[str, Any]] = []
        failed_cases: list[dict[str, Any]] = []

        for tc in cases:
            cat_stats = by_category.setdefault(tc.category, {"total": 0, "passed": 0})
            cat_stats["total"] += 1

            t0 = time.perf_counter()
            # Retrieve across target document if specified in query/case, else all loaded docs
            target_pipes = (
                [pipelines[tc.expected_document]]
                if tc.expected_document and tc.expected_document in pipelines and len(pipelines) == 1
                else list(pipelines.values())
            )

            all_hits = []
            for p in target_pipes:
                hits = p.retriever.retrieve(
                    tc.query,
                    top_k=self.top_k,
                    retrieval_mode=retrieval_mode,
                    min_score=0.0,
                )
                all_hits.extend(hits)
            all_hits.sort(key=lambda x: x[1], reverse=True)
            top_hits = all_hits[: self.top_k]

            # Extract & validate from top hits
            specs = []
            if top_hits:
                first_pipe = target_pipes[0]
                raw_specs = first_pipe.extractor.extract(tc.query, top_hits)
                for s in raw_specs:
                    src_pipe = pipelines.get(s.source_document, first_pipe)
                    specs.append(
                        src_pipe.validator.validate(
                            s,
                            page_text_map=src_pipe.page_text_map or None,
                            query=tc.query,
                            expected_document=s.source_document,
                        )
                    )

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            if not tc.is_answerable:
                # Filter out specs explicitly rejected as unsupported/insufficient
                active_specs = [
                    s for s in specs
                    if s.verification_status in (VerificationStatus.VERIFIED.value, VerificationStatus.PARTIALLY_SUPPORTED.value)
                ]
                rejected = len(active_specs) == 0
                if rejected:
                    neg_rejected_count += 1
                    cat_stats["passed"] += 1
                    outcome = "PASS_REJECTED"
                else:
                    neg_hallucinated_count += 1
                    outcome = "FAIL_HALLUCINATED"
                    failed_cases.append({
                        "id": tc.id,
                        "category": tc.category,
                        "query": tc.query,
                        "expected": "ABSTAIN / NOT_FOUND",
                        "actual": f"{active_specs[0].component}: {active_specs[0].value} {active_specs[0].unit}",
                        "reason": "Returned specification for unanswerable/adversarial query",
                    })
                if include_details:
                    details.append({
                        "id": tc.id,
                        "split": tc.split,
                        "category": tc.category,
                        "query": tc.query,
                        "outcome": outcome,
                        "latency_ms": round(latency_ms, 2),
                    })
                continue

            # Retrieval IR metrics
            rel_flags = [1 if _is_chunk_relevant(ch, tc) else 0 for ch, _ in top_hits]
            hit_any = 1.0 if any(rel_flags) else 0.0
            recalls.append(hit_any)
            precisions.append(sum(rel_flags) / max(1, len(top_hits)))

            rr = 0.0
            for rank, flag in enumerate(rel_flags, start=1):
                if flag:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)

            dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(rel_flags, start=1))
            ideal = sorted(rel_flags, reverse=True)
            idcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(ideal, start=1))
            ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)

            top_spec = specs[0] if specs else None
            comp_ok = bool(top_spec and _component_matches(tc.expected_component, top_spec.component))

            val_ok = False
            if top_spec and tc.expected_value:
                val_ok = values_equivalent(
                    tc.expected_value, tc.expected_unit, top_spec.value, top_spec.unit
                )
                if not val_ok and top_spec.alternate_value:
                    val_ok = values_equivalent(
                        tc.expected_value, tc.expected_unit, top_spec.alternate_value, top_spec.alternate_unit
                    )
                if not val_ok and top_spec.part_number and tc.expected_unit == "part_number":
                    val_ok = tc.expected_value.lower() in top_spec.part_number.lower()

            unit_ok = bool(
                top_spec and _unit_matches(tc.expected_unit, top_spec.unit, top_spec.alternate_unit)
            )
            citation_ok = bool(
                top_spec
                and (not tc.expected_page or top_spec.page_number == tc.expected_page)
                and (not tc.expected_document or top_spec.source_document == tc.expected_document)
            )
            grounded_ok = bool(
                top_spec
                and top_spec.verification_status in (
                    VerificationStatus.VERIFIED.value,
                    VerificationStatus.PARTIALLY_SUPPORTED.value,
                )
            )

            if comp_ok:
                comp_ok_count += 1
            if val_ok:
                val_ok_count += 1
            if unit_ok:
                unit_ok_count += 1
            if citation_ok:
                citation_ok_count += 1
            if grounded_ok:
                grounded_ok_count += 1

            passed = comp_ok and val_ok and unit_ok and citation_ok and grounded_ok
            if passed:
                cat_stats["passed"] += 1
            else:
                if top_spec is not None and (not val_ok or not citation_ok or not unit_ok):
                    answerable_grounding_failures += 1
                failed_cases.append({
                    "id": tc.id,
                    "category": tc.category,
                    "query": tc.query,
                    "expected": f"{tc.expected_value} {tc.expected_unit or ''} (p.{tc.expected_page}, {tc.expected_document})",
                    "actual": (
                        f"{top_spec.value} {top_spec.unit} (p.{top_spec.page_number}, {top_spec.source_document}) [{top_spec.verification_status}]"
                        if top_spec
                        else "None (abstained)"
                    ),
                    "reason": (
                        "Missed retrieval/extraction"
                        if not top_spec
                        else f"comp={comp_ok}, val={val_ok}, unit={unit_ok}, cite={citation_ok}, grounded={grounded_ok}"
                    ),
                })

            if include_details:
                details.append({
                    "id": tc.id,
                    "split": tc.split,
                    "category": tc.category,
                    "query": tc.query,
                    "outcome": "PASS" if passed else "FAIL",
                    "top_val": f"{top_spec.value} {top_spec.unit}" if top_spec else "None",
                    "expected_val": f"{tc.expected_value} {tc.expected_unit or ''}".strip(),
                    "top_page": top_spec.page_number if top_spec else None,
                    "expected_page": tc.expected_page,
                    "verification_status": top_spec.verification_status if top_spec else "not_found",
                    "latency_ms": round(latency_ms, 2),
                })

        n_ans = max(1, len(answerable_cases))
        n_neg = len(negative_cases)
        n_tot = max(1, len(cases))

        hallucination_rate = (answerable_grounding_failures + neg_hallucinated_count) / n_tot

        category_accuracy = {
            cat: {
                "total": st["total"],
                "passed": st["passed"],
                "accuracy": round(st["passed"] / max(1, st["total"]), 4),
            }
            for cat, st in sorted(by_category.items())
        }

        return {
            "retrieval_mode": retrieval_mode,
            "total_cases": len(cases),
            "answerable_cases": len(answerable_cases),
            "negative_cases": n_neg,
            "metrics": {
                "precision_at_k": round(sum(precisions) / max(1, len(precisions)), 4),
                "recall_at_k": round(sum(recalls) / max(1, len(recalls)), 4),
                "mrr": round(sum(rrs) / max(1, len(rrs)), 4),
                "ndcg_at_k": round(sum(ndcgs) / max(1, len(ndcgs)), 4),
                "exact_value_match_accuracy": round(val_ok_count / n_ans, 4),
                "component_accuracy": round(comp_ok_count / n_ans, 4),
                "unit_accuracy": round(unit_ok_count / n_ans, 4),
                "citation_accuracy": round(citation_ok_count / n_ans, 4),
                "evidence_grounding_rate": round(grounded_ok_count / n_ans, 4),
                "hallucination_rate": round(hallucination_rate, 4),
                "abstention_accuracy": round(neg_rejected_count / max(1, n_neg), 4) if n_neg else 1.0,
                "negative_rejection_rate": round(neg_rejected_count / max(1, n_neg), 4) if n_neg else 1.0,
                "p50_latency_ms": _percentile(latencies, 50),
                "p95_latency_ms": _percentile(latencies, 95),
            },
            "by_category": category_accuracy,
            "failed_queries": failed_cases,
            "details": details if include_details else [],
        }


def run_full_benchmark_suite(
    dataset_path: Path,
    manual_paths: dict[str, Path],
    use_mock_embedder: bool = False,
    top_k: int = 5,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run complete tuning + held-out evaluation and 4-mode ablation study."""
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cases = [BenchmarkCase(**item) for item in raw_data.get("cases", [])]
    embedder = DeterministicMockEmbedder(384) if use_mock_embedder else get_embedder()

    tracemalloc.start()
    ingest_t0 = time.perf_counter()

    pipelines: dict[str, AutoSpecPipeline] = {}
    for doc_name, pdf_path in manual_paths.items():
        if not pdf_path.exists():
            continue
        pipe = AutoSpecPipeline(embedder=embedder, llm_provider="none")
        pipe.ingest_file(pdf_path, force_rebuild=True)
        pipelines[doc_name] = pipe

    ingestion_time_ms = round((time.perf_counter() - ingest_t0) * 1000, 2)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = round(peak_bytes / (1024 * 1024), 2)

    runner = EvaluationRunner(embedder=embedder, top_k=top_k)

    tuning_cases = [c for c in cases if c.split == "tuning"]
    held_out_cases = [c for c in cases if c.split == "held_out"]

    # Run ablation across all 4 strategies on held_out (or all cases if no held_out)
    ablation_target = held_out_cases or cases
    ablation_modes: list[RetrievalMode] = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    ablation_results: dict[str, Any] = {}
    for mode in ablation_modes:
        res = runner.evaluate_cases(pipelines, ablation_target, retrieval_mode=mode, include_details=False)
        ablation_results[mode] = res["metrics"]

    full_eval = runner.evaluate_cases(pipelines, cases, retrieval_mode="hybrid_rerank", include_details=True)
    tuning_eval = runner.evaluate_cases(pipelines, tuning_cases, retrieval_mode="hybrid_rerank", include_details=False) if tuning_cases else None
    held_out_eval = runner.evaluate_cases(pipelines, held_out_cases, retrieval_mode="hybrid_rerank", include_details=False) if held_out_cases else None

    report: dict[str, Any] = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "embedder": "DeterministicMockEmbedder(384)" if use_mock_embedder else getattr(embedder, "model_name", "SentenceTransformer"),
            "top_k": top_k,
            "llm_provider": "none (deterministic offline extractor)",
            "documents_indexed": list(pipelines.keys()),
            "dataset_path": str(dataset_path),
            "total_cases": len(cases),
            "tuning_cases": len(tuning_cases),
            "held_out_cases": len(held_out_cases),
        },
        "system_performance": {
            "ingestion_time_ms": ingestion_time_ms,
            "peak_traced_memory_mb": peak_memory_mb,
        },
        "overall": full_eval,
        "tuning_split": tuning_eval["metrics"] if tuning_eval else None,
        "held_out_split": held_out_eval["metrics"] if held_out_eval else None,
        "ablation_study": ablation_results,
        "limitations": (
            "Benchmark evaluated against self-created OEM-style multi-vehicle manuals (Apex Motors M400, S250, and unseen R180) "
            "plus synthetic rotated/adversarial pages. External proprietary OEM PDFs are not bundled due to copyright."
        ),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
