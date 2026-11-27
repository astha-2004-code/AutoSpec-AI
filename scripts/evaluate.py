"""Empirical benchmark and ablation evaluation runner for AutoSpec AI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autospec.demo_manuals import ensure_demo_manuals
from autospec.evaluation import run_full_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoSpec AI Empirical Benchmark & Ablation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmarks/autospec_benchmark_v2.json",
        help="Path to benchmark dataset JSON",
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock embedder")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval depth")
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmarks/latest_benchmark.json",
        help="Output path for measured benchmark report JSON",
    )
    args = parser.parse_args()

    demo_manuals = ensure_demo_manuals(Path("data/demo"))
    manual_paths: dict[str, Path] = {spec.filename: path for spec, path in demo_manuals}

    r180 = Path("data/benchmarks/2024_Apex_R180_Cargo_Van_Workshop_Manual.pdf")
    if r180.exists():
        manual_paths[r180.name] = r180

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        from scripts.build_benchmark_dataset import generate_benchmark_dataset

        dataset_path = generate_benchmark_dataset()

    print("\n" + "=" * 72)
    print("AutoSpec AI — Empirical Benchmark & Ablation Suite")
    print("=" * 72)

    report = run_full_benchmark_suite(
        dataset_path=dataset_path,
        manual_paths=manual_paths,
        use_mock_embedder=args.mock,
        top_k=args.top_k,
        output_path=Path(args.output),
    )

    # Also mirror to storage/benchmarks/latest_benchmark.json
    storage_out = Path("storage/benchmarks/latest_benchmark.json")
    storage_out.parent.mkdir(parents=True, exist_ok=True)
    storage_out.write_text(Path(args.output).read_text(encoding="utf-8"), encoding="utf-8")

    print("\nOverall Metrics (hybrid_rerank):")
    print(json.dumps(report["overall"]["metrics"], indent=2))
    print("\nAblation Study (Held-Out Split):")
    print(json.dumps(report["ablation_study"], indent=2))
    print("\nCategory Breakdown:")
    print(json.dumps(report["overall"]["by_category"], indent=2))
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
