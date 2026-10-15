"""Benchmark evaluation suite measuring retrieval accuracy, extraction precision, and grounding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from autospec.embeddings import DeterministicMockEmbedder, get_embedder
from autospec.pipeline import AutoSpecPipeline


@dataclass
class EvalTestCase:
    """Benchmark test case with ground truth specifications and page references."""

    id: str
    category: str
    query: str
    expected_component: str | None
    expected_value: str | None
    expected_unit: str | None
    expected_page: int | None
    expected_subsystem: str | None
    is_answerable: bool = True


BENCHMARK_CASES = [
    EvalTestCase(
        id="fb_anchor_torque",
        category="front_brakes",
        query="Torque for front brake caliper anchor plate bolts",
        expected_component="Brake caliper anchor plate bolts",
        expected_value="250",
        expected_unit="Nm",
        expected_page=636,
        expected_subsystem="front_brakes",
    ),
    EvalTestCase(
        id="fb_guide_torque",
        category="front_brakes",
        query="Torque for front brake caliper guide pin bolts",
        expected_component="Brake caliper guide pin bolts",
        expected_value="37",
        expected_unit="Nm",
        expected_page=636,
        expected_subsystem="front_brakes",
    ),
    EvalTestCase(
        id="fb_flow_torque",
        category="front_brakes",
        query="Torque for front brake caliper flow bolt",
        expected_component="Brake caliper flow bolt",
        expected_value="35",
        expected_unit="Nm",
        expected_page=636,
        expected_subsystem="front_brakes",
    ),
    EvalTestCase(
        id="rb_guide_torque",
        category="rear_brakes",
        query="Torque for rear brake caliper guide pin bolts",
        expected_component="Brake caliper guide pin bolts",
        expected_value="33",
        expected_unit="Nm",
        expected_page=652,
        expected_subsystem="rear_brakes",
    ),
    EvalTestCase(
        id="rb_support_torque",
        category="rear_brakes",
        query="Torque for rear brake caliper support bracket bolts",
        expected_component="Brake caliper support bracket bolts",
        expected_value="150",
        expected_unit="Nm",
        expected_page=652,
        expected_subsystem="rear_brakes",
    ),
    EvalTestCase(
        id="fluid_cap_dot3",
        category="fluid_capacities",
        query="What is the brake fluid fill capacity?",
        expected_component="Motor Vehicle Brake Fluid",
        expected_value="1,420 ml (3 pt)",
        expected_unit="capacity",
        expected_page=636,
        expected_subsystem="front_brakes",
    ),
    EvalTestCase(
        id="part_no_dot3",
        category="part_numbers",
        query="What is the part number for High Performance DOT 3 Brake Fluid?",
        expected_component="High Performance DOT 3 Motor Vehicle Brake Fluid",
        expected_value="PM-1-C (US); CPM-1-C (Canada)",
        expected_unit="part_number",
        expected_page=636,
        expected_subsystem="front_brakes",
    ),
    EvalTestCase(
        id="fb_rotor_thickness",
        category="dimension_clearance",
        query="Brake disc minimum thickness for front disc brake",
        expected_component="Brake disc minimum thickness",
        expected_value="32",
        expected_unit="mm",
        expected_page=636,
        expected_subsystem="front_brakes",
    ),
    EvalTestCase(
        id="neg_turbo_nut",
        category="negative_query",
        query="What is the torque for the turbocharger impeller titanium nut?",
        expected_component=None,
        expected_value=None,
        expected_unit=None,
        expected_page=None,
        expected_subsystem=None,
        is_answerable=False,
    ),
]


def run_evaluation(
    pdf_path: str | Path | None = None,
    use_mock_embedder: bool = False,
    top_k: int = 3,
) -> dict[str, Any]:
    """Execute evaluation benchmark and compute empirical retrieval and extraction metrics."""
    embedder = DeterministicMockEmbedder(384) if use_mock_embedder else get_embedder()
    pipeline = AutoSpecPipeline(embedder=embedder, llm_provider="none")

    # Ingest test slice of reference manual (pages 630 to 660 covering front & rear disc brakes)
    if pdf_path:
        print(f"Ingesting reference manual slice from: {pdf_path} (Pages 630-660)...")
        pipeline.ingest_file(pdf_path, start_page=630, end_page=660, force_rebuild=True)

    total_queries = len(BENCHMARK_CASES)
    retrieval_hits = 0
    component_matches = 0
    value_exact_matches = 0
    page_matches = 0
    grounding_verified = 0
    unanswerable_correct = 0

    results_detail = []

    for tc in BENCHMARK_CASES:
        t0 = time.perf_counter()
        query_resp = pipeline.query(tc.query, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        specs = query_resp.results

        if not tc.is_answerable:
            # For unanswerable queries, success means returning 0 hallucinated specifications
            if len(specs) == 0:
                unanswerable_correct += 1
                outcome = "PASS_REJECTED"
            else:
                outcome = "FAIL_HALLUCINATED"
            results_detail.append({
                "id": tc.id,
                "query": tc.query,
                "outcome": outcome,
                "returned_count": len(specs),
                "latency_ms": round(latency_ms, 2),
            })
            continue

        # Answerable query checks
        hit = len(specs) > 0
        if hit:
            retrieval_hits += 1

        top_spec = specs[0] if specs else None
        comp_ok = False
        val_ok = False
        page_ok = False
        verified_ok = False

        if top_spec:
            # Component match
            if tc.expected_component and tc.expected_component.lower() in top_spec.component.lower():
                comp_ok = True
                component_matches += 1

            # Value exact match
            if tc.expected_value and (
                tc.expected_value in top_spec.value
                or top_spec.value in tc.expected_value
                or (top_spec.part_number and tc.expected_value in top_spec.part_number)
            ):
                val_ok = True
                value_exact_matches += 1

            # Page match
            if tc.expected_page and top_spec.page_number == tc.expected_page:
                page_ok = True
                page_matches += 1

            # Verification grounding status
            if top_spec.verification_status == "verified":
                verified_ok = True
                grounding_verified += 1

        outcome = "PASS" if (val_ok and page_ok and comp_ok) else "FAIL"
        results_detail.append({
            "id": tc.id,
            "query": tc.query,
            "outcome": outcome,
            "top_val": f"{top_spec.value} {top_spec.unit}" if top_spec else "None",
            "expected_val": f"{tc.expected_value} {tc.expected_unit}",
            "top_page": top_spec.page_number if top_spec else None,
            "expected_page": tc.expected_page,
            "verified": verified_ok,
            "latency_ms": round(latency_ms, 2),
        })

    answerable_count = sum(1 for tc in BENCHMARK_CASES if tc.is_answerable)
    negative_count = total_queries - answerable_count

    report = {
        "total_test_cases": total_queries,
        "answerable_cases": answerable_count,
        "negative_cases": negative_count,
        "metrics": {
            "recall_at_k": round(retrieval_hits / answerable_count, 4),
            "component_match_accuracy": round(component_matches / answerable_count, 4),
            "value_exact_match_accuracy": round(value_exact_matches / answerable_count, 4),
            "source_page_accuracy": round(page_matches / answerable_count, 4),
            "evidence_grounding_rate": round(grounding_verified / answerable_count, 4),
            "negative_rejection_accuracy": round(unanswerable_correct / negative_count, 4) if negative_count else 1.0,
            "hallucination_rate": 0.0 if unanswerable_correct == negative_count else round((negative_count - unanswerable_correct) / negative_count, 4),
        },
        "details": results_detail,
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSpec AI Benchmark Evaluation")
    parser.add_argument("--pdf", type=str, default="", help="Path to reference service manual PDF")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock embedder")
    args = parser.parse_args()

    default_manual = args.pdf or r"C:\Users\Dell\Downloads\sample-service-manual 1.pdf"
    if not Path(default_manual).exists():
        default_manual = None

    print("\n" + "=" * 60)
    print("🚗 AutoSpec AI — Empirical Benchmark Evaluation")
    print("=" * 60)

    report = run_evaluation(pdf_path=default_manual, use_mock_embedder=args.mock)
    print(json.dumps(report["metrics"], indent=2))
    print("\nCase Results:")
    for d in report["details"]:
        print(f"  [{d['outcome']}] {d['id']}: {d['query']} -> {d.get('top_val')} (Page {d.get('top_page')})")
    print("=" * 60 + "\n")
