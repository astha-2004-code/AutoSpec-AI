"""CLI ingestion tool for AutoSpec AI service manuals."""

from __future__ import annotations

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from autospec.pipeline import AutoSpecPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest workshop manual PDF into AutoSpec AI")
    parser.add_argument("--pdf", type=str, required=True, help="Path to local workshop manual PDF")
    parser.add_argument("--start", type=int, default=None, help="Start page (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="End page (1-indexed)")
    parser.add_argument("--force", action="store_true", help="Force rebuild index, ignoring cache")

    args = parser.parse_args()

    print("\n🚗 AutoSpec AI — Ingestion Pipeline")
    print(f"Target Manual: {args.pdf}")
    if args.start or args.end:
        print(f"Page Range: {args.start or 1} to {args.end or 'End'}")
    print(f"Force Rebuild: {args.force}\n")

    pipeline = AutoSpecPipeline()

    def progress_callback(cur, total):
        pct = int((cur / total) * 100)
        sys.stdout.write(f"\rProcessing page {cur}/{total} [{pct}%]...")
        sys.stdout.flush()

    t0 = time.perf_counter()
    resp = pipeline.ingest_file(
        pdf_path_str=args.pdf,
        start_page=args.start,
        end_page=args.end,
        force_rebuild=args.force,
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n\n✅ Ingestion Status: {resp.status.upper()}")
    print(f"Document: {resp.filename}")
    print(f"Fingerprint: {resp.fingerprint}")
    print(f"Extracted Chunks: {resp.chunk_count}")
    print(f"Cached Index: {resp.is_cached}")
    print(f"Total Time: {elapsed:.2f} s ({resp.processing_time_ms} ms)")


if __name__ == "__main__":
    main()
