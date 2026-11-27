import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertCircle, Download, Info } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/Button";

export function AnalyticsPage() {
  const {
    data: report,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["benchmark-full"],
    queryFn: api.getBenchmark,
  });

  if (isLoading) {
    return (
      <div className="mx-auto max-w-[1320px] space-y-4 px-4 py-8 sm:px-6">
        <div className="h-8 w-64 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-24 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="mx-auto max-w-xl px-4 py-16 text-center">
        <AlertCircle className="mx-auto h-8 w-8 text-red-500" />
        <h1 className="mt-3 text-base font-semibold">Benchmark Report Unavailable</h1>
        <p className="mt-1 text-xs text-slate-500">
          Run <code className="font-mono">python scripts/evaluate.py</code> to generate measured
          evaluation data.
        </p>
      </div>
    );
  }

  const m = report.overall.metrics;
  const ablationData = Object.entries(report.ablation_study).map(([mode, stats]) => ({
    mode,
    Recall: Number((stats.recall_at_k * 100).toFixed(1)),
    MRR: Number((stats.mrr * 100).toFixed(1)),
    "Exact Value": Number((stats.exact_value_match_accuracy * 100).toFixed(1)),
    "Citation Acc": Number((stats.citation_accuracy * 100).toFixed(1)),
  }));

  const categoryData = Object.entries(report.overall.by_category).map(([cat, stats]) => ({
    category: cat.replace(/_/g, " "),
    accuracy: Number((stats.accuracy * 100).toFixed(1)),
    passed: stats.passed,
    total: stats.total,
  }));

  const downloadReport = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "autospec_benchmark_report.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mx-auto max-w-[1320px] space-y-8 px-4 py-8 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Evaluation & Retrieval Analytics
          </h1>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Measured on {new Date(report.evaluated_at).toLocaleString()} •{" "}
            {report.configuration.total_cases} verified queries ({report.configuration.tuning_cases}{" "}
            tuning / {report.configuration.held_out_cases} held-out) • Embedder:{" "}
            <code className="font-mono">{report.configuration.embedder}</code>
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={downloadReport}>
          <Download className="h-3.5 w-3.5" /> Download Benchmark JSON
        </Button>
      </div>

      {/* Primary KPIs */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[
          { label: "Precision@5", val: `${(m.precision_at_k * 100).toFixed(1)}%` },
          { label: "Recall@5", val: `${(m.recall_at_k * 100).toFixed(1)}%` },
          { label: "MRR / nDCG@5", val: `${m.mrr.toFixed(3)} / ${m.ndcg_at_k.toFixed(3)}` },
          {
            label: "Value / Unit Acc",
            val: `${(m.exact_value_match_accuracy * 100).toFixed(1)}% / ${(m.unit_accuracy * 100).toFixed(1)}%`,
          },
          {
            label: "Hallucination / Abstention",
            val: `${(m.hallucination_rate * 100).toFixed(2)}% / ${(m.abstention_accuracy * 100).toFixed(1)}%`,
          },
          { label: "Latency P50 / P95", val: `${m.p50_latency_ms} / ${m.p95_latency_ms} ms` },
        ].map((k) => (
          <div
            key={k.label}
            className="rounded-lg border border-slate-200 bg-white p-3.5 dark:border-slate-800 dark:bg-[#11161D]"
          >
            <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
              {k.label}
            </p>
            <p className="mt-1 font-mono text-base font-semibold text-slate-900 dark:text-white">
              {k.val}
            </p>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            Retrieval Ablation Study (Held-Out Split, %)
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Comparing BM25 only, Dense only, Hybrid RRF, and Hybrid RRF + Reranking.
          </p>
          <div className="mt-4 h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ablationData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis dataKey="mode" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Recall" fill="#3B82F6" radius={[2, 2, 0, 0]} />
                <Bar dataKey="MRR" fill="#0EA5E9" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Exact Value" fill="#10B981" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            Accuracy by Question Category (%)
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            End-to-end strict component, normalized value, unit, and source page pass rate.
          </p>
          <div className="mt-4 h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.25} />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
                <YAxis dataKey="category" type="category" width={140} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="accuracy" name="Pass Rate (%)" fill="#2563EB" radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Methodology & Limitations */}
      <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
        <div className="flex items-start gap-2.5">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-accent-600 dark:text-accent-500" />
          <div className="text-xs leading-relaxed text-slate-600 dark:text-slate-400">
            <p className="font-semibold text-slate-900 dark:text-white">
              Methodology, System Telemetry & Benchmark Scope
            </p>
            <p className="mt-1">
              Ingestion Time:{" "}
              <strong className="font-mono text-slate-800 dark:text-slate-200">
                {report.system_performance.ingestion_time_ms} ms
              </strong>{" "}
              • Peak Traced Memory:{" "}
              <strong className="font-mono text-slate-800 dark:text-slate-200">
                {report.system_performance.peak_traced_memory_mb} MB
              </strong>{" "}
              • Evidence Grounding Rate:{" "}
              <strong className="font-mono text-slate-800 dark:text-slate-200">
                {(m.evidence_grounding_rate * 100).toFixed(1)}%
              </strong>
              .
            </p>
            <p className="mt-1">{report.limitations}</p>
          </div>
        </div>
      </div>

      {/* Failed Query Inspection */}
      <div className="rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#11161D]">
        <div className="border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            Failed Query Inspection ({report.overall.failed_queries.length})
          </h2>
        </div>
        <div className="max-h-96 overflow-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase text-slate-500 dark:border-slate-800 dark:bg-[#0F151D]">
                <th className="px-4 py-2.5">ID / Category</th>
                <th className="px-4 py-2.5">Query</th>
                <th className="px-4 py-2.5">Expected</th>
                <th className="px-4 py-2.5">Actual</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {report.overall.failed_queries.slice(0, 50).map((f) => (
                <tr key={f.id}>
                  <td className="px-4 py-2.5 font-mono text-[11px]">
                    <p>{f.id}</p>
                    <p className="text-slate-500">{f.category}</p>
                  </td>
                  <td className="px-4 py-2.5 text-slate-800 dark:text-slate-200">{f.query}</td>
                  <td className="px-4 py-2.5 font-mono text-[11px] text-emerald-600 dark:text-emerald-400">
                    {f.expected}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-[11px] text-rose-600 dark:text-rose-400">
                    {f.actual}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
