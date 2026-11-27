import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  CheckCircle2,
  Database,
  FileSearch,
  GitCompare,
  Github,
  Layers,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { VerificationBadge } from "@/components/ui/VerificationBadge";
import type { QueryResponse } from "@/types/api";

const DEMO_QUERIES = [
  {
    label: "Front Guide Pin Torque",
    query: "What is the torque for front brake caliper guide pin bolts on the 2022 Apex M400?",
  },
  {
    label: "Rear Guide Pin Torque",
    query: "What is the torque for rear brake caliper guide pin bolts on the 2022 Apex M400?",
  },
  {
    label: "Hybrid HV Battery Voltage",
    query: "What is the high-voltage traction battery nominal voltage on 2023 Apex S250?",
  },
  {
    label: "Negative Hallucination Test",
    query: "What is the torque for the turbocharger impeller titanium locknut?",
  },
];

export function LandingPage() {
  const [activeQuery, setActiveQuery] = useState(DEMO_QUERIES[0].query);

  const { data: benchmark } = useQuery({
    queryKey: ["benchmark-summary"],
    queryFn: api.getBenchmark,
    retry: 1,
  });

  const {
    data: demoResult,
    isLoading: demoLoading,
    error: demoError,
  } = useQuery<QueryResponse>({
    queryKey: ["landing-demo-query", activeQuery],
    queryFn: () =>
      api.querySpecs({
        query: activeQuery,
        top_k: 2,
        retrieval_mode: "hybrid_rerank",
      }),
    retry: 1,
  });

  const metrics = benchmark?.overall?.metrics;

  return (
    <div className="mx-auto max-w-[1320px] px-4 py-12 sm:px-6 lg:py-16">
      {/* Hero */}
      <section className="grid gap-10 lg:grid-cols-12 lg:items-center">
        <div className="lg:col-span-7">
          <div className="inline-flex items-center gap-2 rounded border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 dark:border-slate-800 dark:bg-[#11161D] dark:text-slate-300">
            <ShieldCheck className="h-3.5 w-3.5 text-accent-600 dark:text-accent-500" />
            Evidence-Grounded Automotive RAG Platform
          </div>

          <h1 className="mt-4 text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl lg:text-[42px] lg:leading-[1.15] dark:text-white">
            Engineering Specifications.
            <br />
            Verified at the Source.
          </h1>

          <p className="mt-4 max-w-2xl text-base leading-relaxed text-slate-600 dark:text-slate-400">
            Generic document assistants conflate front and rear assemblies, drop alternate units,
            and fabricate fastener torque values. AutoSpec AI extracts structured automotive
            specifications with strict unit verification, subsystem disambiguation, and
            coordinate-linked PDF evidence.
          </p>

          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Link to="/overview">
              <Button size="lg">
                Launch Workspace
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link to="/workspace">
              <Button variant="outline" size="lg">
                Try Live Demo
              </Button>
            </Link>
            <Link to="/documents">
              <Button variant="outline" size="lg">
                <Upload className="h-4 w-4" />
                Upload Manual
              </Button>
            </Link>
            <a
              href="https://github.com/astha-2004-code/AutoSpec-AI"
              target="_blank"
              rel="noreferrer"
            >
              <Button variant="ghost" size="lg">
                <Github className="h-4 w-4" />
                View Source
              </Button>
            </a>
          </div>

          <p className="mt-3 text-xs text-slate-500 dark:text-slate-500">
            No account required. Pre-loaded with two multi-system demonstration workshop manuals
            (2022 Apex M400 & 2023 Apex S250 Hybrid).
          </p>
        </div>

        {/* Interactive Live Product Demonstration */}
        <div className="lg:col-span-5">
          <div className="rounded-lg border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-[#11161D]">
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Interactive Verification Demo
              </span>
              <span className="font-mono text-[11px] text-emerald-600 dark:text-emerald-400">
                Deterministic Offline Mode
              </span>
            </div>

            <div className="p-4">
              <div className="flex flex-wrap gap-1.5">
                {DEMO_QUERIES.map((dq) => (
                  <button
                    key={dq.label}
                    onClick={() => setActiveQuery(dq.query)}
                    className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors ${
                      activeQuery === dq.query
                        ? "border-accent-600 bg-accent-600/10 text-accent-600 dark:text-accent-400"
                        : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-800 dark:text-slate-400 dark:hover:bg-slate-800/60"
                    }`}
                  >
                    {dq.label}
                  </button>
                ))}
              </div>

              <div className="mt-3 rounded border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs text-slate-800 dark:border-slate-800 dark:bg-[#0B0F14] dark:text-slate-200">
                {activeQuery}
              </div>

              <div className="mt-3 min-h-[150px]">
                {demoLoading && (
                  <div className="space-y-2 py-4">
                    <div className="h-4 w-2/3 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
                    <div className="h-10 w-full animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
                    <div className="h-4 w-1/2 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
                  </div>
                )}

                {demoError && (
                  <p className="py-6 text-center text-xs text-slate-500 dark:text-slate-400">
                    Connect the FastAPI backend on port 8000 to run live interactive queries.
                  </p>
                )}

                {!demoLoading && demoResult && demoResult.results.length === 0 && (
                  <div className="rounded border border-slate-200 bg-slate-50 p-3.5 dark:border-slate-800 dark:bg-[#0B0F14]">
                    <div className="flex items-center justify-between">
                      <VerificationBadge status="not_found" />
                      <span className="font-mono text-[11px] text-slate-500">
                        {demoResult.processing_time_ms.toFixed(1)} ms
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-slate-700 dark:text-slate-300">
                      {demoResult.missing_info_explanation ||
                        "Specification not found in indexed manuals. AutoSpec AI abstained instead of fabricating a value."}
                    </p>
                  </div>
                )}

                {!demoLoading && demoResult && demoResult.results.length > 0 && (
                  <div className="rounded border border-slate-200 bg-slate-50 p-3.5 dark:border-slate-800 dark:bg-[#0B0F14]">
                    <div className="flex items-center justify-between gap-2">
                      <VerificationBadge status={demoResult.results[0].verification_status} />
                      <span className="font-mono text-[11px] text-slate-500">
                        {demoResult.processing_time_ms.toFixed(1)} ms
                      </span>
                    </div>
                    <p className="mt-2 text-sm font-semibold text-slate-900 dark:text-white">
                      {demoResult.results[0].component}
                    </p>
                    <p className="mt-1 font-mono text-lg font-semibold text-accent-600 dark:text-accent-500">
                      {demoResult.results[0].value} {demoResult.results[0].unit}
                      {demoResult.results[0].alternate_value && (
                        <span className="ml-2 text-xs font-normal text-slate-500">
                          ({demoResult.results[0].alternate_value}{" "}
                          {demoResult.results[0].alternate_unit})
                        </span>
                      )}
                    </p>
                    <p className="mt-1.5 text-[11px] text-slate-600 dark:text-slate-400">
                      {demoResult.results[0].section} • PDF Page {demoResult.results[0].page_number}
                      {demoResult.results[0].printed_page_label
                        ? ` (Manual ${demoResult.results[0].printed_page_label})`
                        : ""}
                    </p>
                    <p className="mt-2 border-l-2 border-accent-600 pl-2 font-mono text-[11px] text-slate-600 dark:text-slate-400">
                      "{demoResult.results[0].source_quote}"
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Real Measured Evaluation Metrics */}
      {metrics && (
        <section className="mt-14 border-t border-slate-200 pt-10 dark:border-slate-800">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                Measured Evaluation Results
              </h2>
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                Reproducible metrics across {benchmark.configuration.total_cases} source-verified
                queries ({benchmark.configuration.embedder}, hybrid + rerank).
              </p>
            </div>
            <Link
              to="/analytics"
              className="text-xs font-medium text-accent-600 hover:underline dark:text-accent-500"
            >
              Inspect Full Ablation & Failed Queries →
            </Link>
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {[
              { label: "Recall@5", val: `${(metrics.recall_at_k * 100).toFixed(1)}%` },
              { label: "MRR", val: metrics.mrr.toFixed(3) },
              {
                label: "Value Accuracy",
                val: `${(metrics.exact_value_match_accuracy * 100).toFixed(1)}%`,
              },
              {
                label: "Citation Accuracy",
                val: `${(metrics.citation_accuracy * 100).toFixed(1)}%`,
              },
              {
                label: "Hallucination Rate",
                val: `${(metrics.hallucination_rate * 100).toFixed(2)}%`,
              },
              { label: "P50 Latency", val: `${metrics.p50_latency_ms.toFixed(1)} ms` },
            ].map((m) => (
              <div
                key={m.label}
                className="rounded-md border border-slate-200 bg-white p-3.5 dark:border-slate-800 dark:bg-[#11161D]"
              >
                <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  {m.label}
                </p>
                <p className="mt-1 font-mono text-xl font-semibold text-slate-900 dark:text-white">
                  {m.val}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Core Engineering Problem & Capabilities */}
      <section className="mt-14 border-t border-slate-200 pt-10 dark:border-slate-800">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
          Built for Safety-Critical Workshop Intelligence
        </h2>
        <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[
            {
              icon: Database,
              title: "Table-Aware Ingestion",
              body: "Parses multi-column torque tables, multi-page continuations, tolerances, and rotated layouts while retaining physical and printed page numbers.",
            },
            {
              icon: Layers,
              title: "Hybrid RRF + Reranking",
              body: "Combines FAISS semantic search, automotive-tokenized BM25Plus, subsystem isolation, and second-stage reranking.",
            },
            {
              icon: CheckCircle2,
              title: "Zero-Hallucination Validator",
              body: "Verifies exact numeric bounds, physical unit compatibility, verbatim quote grounding, and cross-subsystem conflicts before display.",
            },
            {
              icon: GitCompare,
              title: "Multi-Vehicle & PDF Evidence",
              body: "Compare specifications across vehicle platforms with SI normalization and inspect highlighted source coordinates in PDF.js.",
            },
          ].map(({ icon: Icon, title, body }) => (
            <div
              key={title}
              className="rounded-md border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]"
            >
              <Icon className="h-5 w-5 text-accent-600 dark:text-accent-500" />
              <h3 className="mt-3 text-sm font-semibold text-slate-900 dark:text-white">{title}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                {body}
              </p>
            </div>
          ))}
        </div>
      </section>

      <footer className="mt-16 flex flex-wrap items-center justify-between gap-4 border-t border-slate-200 pt-6 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        <div className="flex items-center gap-2">
          <FileSearch className="h-4 w-4 text-accent-600 dark:text-accent-500" />
          <span>
            AutoSpec AI — Verify safety-critical torque values against certified OEM publications
            prior to service.
          </span>
        </div>
        <Link
          to="/architecture"
          className="font-medium text-accent-600 hover:underline dark:text-accent-500"
        >
          Technical Architecture
        </Link>
      </footer>
    </div>
  );
}
