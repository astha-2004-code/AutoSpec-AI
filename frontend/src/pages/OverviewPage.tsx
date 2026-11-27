import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Database,
  FileStack,
  GitCompare,
  History,
  Layers,
  Search,
  ShieldCheck,
  TableProperties,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatSubsystem } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { VerificationBadge } from "@/components/ui/VerificationBadge";

export function OverviewPage() {
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ["overview-stats"],
    queryFn: api.getOverview,
  });

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
  });

  const { data: history = [] } = useQuery({
    queryKey: ["history", 6],
    queryFn: () => api.getHistory(6),
  });

  const { data: benchmark } = useQuery({
    queryKey: ["benchmark-summary"],
    queryFn: api.getBenchmark,
    retry: 1,
  });

  return (
    <div className="mx-auto max-w-[1360px] space-y-6 px-4 py-6 sm:px-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Workspace Overview
          </h1>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Live database telemetry, indexed automotive manuals, and verification diagnostics for
            your active session.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link to="/documents">
            <Button variant="outline" size="sm">
              <FileStack className="h-3.5 w-3.5" />
              Upload Manual
            </Button>
          </Link>
          <Link to="/workspace">
            <Button size="sm">
              <Search className="h-3.5 w-3.5" />
              Ask AutoSpec
            </Button>
          </Link>
        </div>
      </div>

      {/* KPI Cards (Strictly Real DB Telemetry) */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          {
            label: "Indexed Manuals",
            value: overviewLoading ? "..." : String(overview?.document_count ?? 0),
            sub: overview
              ? `${overview.demo_document_count} demo • ${overview.user_document_count} session`
              : "Loading workspace...",
            icon: FileStack,
          },
          {
            label: "Pages & Chunks",
            value: overviewLoading ? "..." : `${overview?.total_pages ?? 0} pp`,
            sub: overview ? `${overview.total_chunks} table-aware chunks` : "",
            icon: Layers,
          },
          {
            label: "Extracted Specs",
            value: overviewLoading ? "..." : String(overview?.total_specifications ?? 0),
            sub: overview ? `${overview.verified_specifications} verified records in DB` : "",
            icon: CheckCircle2,
          },
          {
            label: "Session Queries",
            value: overviewLoading ? "..." : String(overview?.query_count ?? 0),
            sub: health
              ? `${health.database_backend?.toUpperCase() || "DB"}${
                  health.pgvector_enabled ? " + pgvector" : ""
                }`
              : "Connecting...",
            icon: Database,
          },
        ].map(({ label, value, sub, icon: Icon }) => (
          <div
            key={label}
            className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#11161D]"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
                {label}
              </span>
              <Icon className="h-4 w-4 text-accent-600 dark:text-accent-500" />
            </div>
            <p className="mt-2 font-mono text-2xl font-semibold text-slate-900 dark:text-white">
              {value}
            </p>
            <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">{sub}</p>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-12">
        {/* Subsystems & Quick Launch */}
        <div className="space-y-6 lg:col-span-7">
          <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
              Quick Engineering Actions
            </h2>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {[
                {
                  to: "/workspace",
                  title: "Ask AutoSpec Workspace",
                  desc: "Multi-manual retrieval with live PDF.js evidence inspector.",
                  icon: Search,
                },
                {
                  to: "/explorer",
                  title: "Specification Explorer",
                  desc: "Filter verified torque & fluid specs and export CSV/JSON.",
                  icon: TableProperties,
                },
                {
                  to: "/compare",
                  title: "Vehicle Comparison",
                  desc: "Cross-check unit-normalized assemblies across vehicles.",
                  icon: GitCompare,
                },
                {
                  to: "/analytics",
                  title: "Evaluation Benchmark",
                  desc: benchmark
                    ? `${benchmark.overall.total_cases}-case source-verified ablation suite.`
                    : "Inspect measured retrieval accuracy & latency telemetry.",
                  icon: Activity,
                },
              ].map(({ to, title, desc, icon: Icon }) => (
                <Link
                  key={to}
                  to={to}
                  className="group rounded-md border border-slate-200 p-3.5 transition-colors hover:border-accent-500/60 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-900/50"
                >
                  <div className="flex items-center justify-between">
                    <Icon className="h-4 w-4 text-accent-600 dark:text-accent-500" />
                    <ArrowRight className="h-3.5 w-3.5 text-slate-400 transition-transform group-hover:translate-x-0.5" />
                  </div>
                  <h3 className="mt-2.5 text-xs font-semibold text-slate-900 dark:text-white">
                    {title}
                  </h3>
                  <p className="mt-1 text-[11px] leading-relaxed text-slate-600 dark:text-slate-400">
                    {desc}
                  </p>
                </Link>
              ))}
            </div>
          </div>

          {/* Indexed Subsystems */}
          <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                Indexed Subsystems Coverage
              </h2>
              <ShieldCheck className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {overview?.subsystems.length ? (
                overview.subsystems.map((sub) => (
                  <Link
                    key={sub}
                    to={`/explorer?subsystem=${encodeURIComponent(sub)}`}
                    className="rounded border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-700 hover:border-accent-500 dark:border-slate-800 dark:bg-[#0B0F14] dark:text-slate-300"
                  >
                    {formatSubsystem(sub)}
                  </Link>
                ))
              ) : (
                <p className="text-xs text-slate-500">No subsystems indexed in current scope.</p>
              )}
            </div>
          </div>
        </div>

        {/* Recent Search History */}
        <div className="rounded-lg border border-slate-200 bg-white p-5 lg:col-span-5 dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white">
              <History className="h-4 w-4 text-slate-500" />
              Recent Queries
            </h2>
            <Link
              to="/history"
              className="text-xs font-medium text-accent-600 hover:underline dark:text-accent-400"
            >
              View all
            </Link>
          </div>

          {history.length === 0 ? (
            <div className="py-10 text-center">
              <p className="text-xs text-slate-500 dark:text-slate-400">
                No specification queries executed in this session yet.
              </p>
              <Link to="/workspace" className="mt-3 inline-block">
                <Button size="sm" variant="outline">
                  Run First Query
                </Button>
              </Link>
            </div>
          ) : (
            <div className="mt-4 divide-y divide-slate-100 dark:divide-slate-800">
              {history.map((item) => (
                <Link
                  key={item.id}
                  to={`/workspace?q=${encodeURIComponent(item.query)}`}
                  className="block py-2.5 hover:bg-slate-50/70 dark:hover:bg-slate-800/30"
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="line-clamp-1 text-xs font-medium text-slate-800 dark:text-slate-200">
                      {item.query}
                    </p>
                    {item.top_status && <VerificationBadge status={item.top_status} />}
                  </div>
                  <p className="mt-1 font-mono text-[10px] text-slate-500">
                    {item.result_count} hits • {item.latency_ms} ms
                  </p>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
