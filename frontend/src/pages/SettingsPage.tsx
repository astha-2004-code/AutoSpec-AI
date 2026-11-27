import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Copy, Database, RefreshCw, Shield, Sliders } from "lucide-react";
import { api, getApiBaseUrl, getSessionId, resetSessionId } from "@/lib/api";
import type { RetrievalMode } from "@/types/api";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/Toast";

export const PREF_UNIT_KEY = "autospec_pref_unit_system";
export const PREF_RETRIEVAL_KEY = "autospec_pref_retrieval_mode";

export function SettingsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [unitSystem, setUnitSystem] = useState<"metric" | "imperial">(
    () => (localStorage.getItem(PREF_UNIT_KEY) as "metric" | "imperial") || "metric"
  );
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>(
    () => (localStorage.getItem(PREF_RETRIEVAL_KEY) as RetrievalMode) || "hybrid_rerank"
  );
  const [sessionId, setSessionId] = useState(() => getSessionId());

  const {
    data: health,
    isLoading,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
  });

  const savePreferences = () => {
    localStorage.setItem(PREF_UNIT_KEY, unitSystem);
    localStorage.setItem(PREF_RETRIEVAL_KEY, retrievalMode);
    toast({
      variant: "success",
      title: "Workspace preferences saved",
      description: `Default unit system set to ${unitSystem}, retrieval mode to ${retrievalMode}.`,
    });
  };

  const handleResetSession = () => {
    const next = resetSessionId();
    setSessionId(next);
    queryClient.invalidateQueries();
    toast({
      variant: "info",
      title: "Session rotated",
      description: "Generated a fresh isolated browser workspace session ID.",
    });
  };

  return (
    <div className="mx-auto max-w-[1100px] space-y-6 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Workspace Settings & Diagnostics
          </h1>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Configure local retrieval defaults and inspect live backend, Neon PostgreSQL, and
            pgvector runtime health.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`} />
          Refresh Health
        </Button>
      </div>

      {/* Live Backend & Database Telemetry */}
      <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-accent-600 dark:text-accent-500" />
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            Runtime Infrastructure Status
          </h2>
        </div>

        {isLoading ? (
          <div className="mt-4 h-24 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
        ) : (
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { label: "API Endpoint", value: getApiBaseUrl() },
              {
                label: "Service Health",
                value: health ? `${health.status.toUpperCase()} (v${health.version})` : "Offline",
              },
              {
                label: "Environment",
                value: health?.environment || "unknown",
              },
              {
                label: "Database Backend",
                value: health
                  ? `${health.database_backend || "sqlite"} (${
                      health.database_ready ? "Connected" : "Degraded"
                    })`
                  : "Unknown",
              },
              {
                label: "Vector Index",
                value: health?.pgvector_enabled
                  ? "PostgreSQL pgvector + FAISS/BM25"
                  : "FAISS + BM25 Hybrid",
              },
              {
                label: "Embedding / Reranker",
                value: health
                  ? `${health.embedding_provider || "onnx"} (${health.embedding_model}) • rerank=${String(health.reranker_enabled)}`
                  : "Unknown",
              },
              {
                label: "Memory RSS",
                value:
                  health && typeof health.memory_rss_mb === "number"
                    ? `${health.memory_rss_mb.toFixed(1)} MB`
                    : "Unknown",
              },
              {
                label: "LLM Provider Mode",
                value: health?.active_llm_provider || "deterministic offline",
              },
              {
                label: "OCR Fallback",
                value: health?.ocr_available ? "Available (pytesseract)" : "Unavailable",
              },
              {
                label: "Demo Collection",
                value: health?.demo_ready
                  ? `Pre-indexed (${health.document_count} docs)`
                  : "Not loaded",
              },
            ].map((row) => (
              <div
                key={row.label}
                className="rounded border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-800 dark:bg-[#0B0F14]"
              >
                <p className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                  {row.label}
                </p>
                <p className="mt-1 truncate font-mono text-xs font-medium text-slate-900 dark:text-white">
                  {row.value}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Engineering Preferences */}
      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex items-center gap-2">
            <Sliders className="h-4 w-4 text-accent-600 dark:text-accent-500" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
              Default Retrieval & Unit Preferences
            </h2>
          </div>

          <div className="mt-4 space-y-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-700 dark:text-slate-300">
                Preferred Comparison Unit System
              </label>
              <select
                value={unitSystem}
                onChange={(e) => setUnitSystem(e.target.value as "metric" | "imperial")}
                className="w-full rounded border border-slate-300 bg-white px-3 py-2 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
              >
                <option value="metric">Metric SI (Nm, L, mm, °C, kPa)</option>
                <option value="imperial">Imperial / US (ft-lb, qt, in, °F, psi)</option>
              </select>
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-700 dark:text-slate-300">
                Default Retrieval Strategy
              </label>
              <select
                value={retrievalMode}
                onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}
                className="w-full rounded border border-slate-300 bg-white px-3 py-2 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
              >
                <option value="hybrid_rerank">Hybrid RRF + Reranking (Recommended)</option>
                <option value="hybrid">Hybrid RRF (Dense + BM25)</option>
                <option value="bm25">Sparse Lexical (BM25)</option>
                <option value="dense">Dense Semantic Vector</option>
              </select>
            </div>

            <Button size="sm" onClick={savePreferences}>
              <CheckCircle2 className="h-3.5 w-3.5" />
              Save Local Preferences
            </Button>
          </div>
        </div>

        {/* Workspace Session Isolation */}
        <div className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-accent-600 dark:text-accent-500" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
              Workspace Session Isolation
            </h2>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
            Uploaded manuals and query logs are scoped strictly to this browser session ID. Public
            demonstration manuals remain readable across all sessions.
          </p>

          <div className="mt-4 rounded border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs text-slate-800 dark:border-slate-800 dark:bg-[#0B0F14] dark:text-slate-200">
            {sessionId}
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                navigator.clipboard.writeText(sessionId);
                toast({ variant: "info", title: "Session ID copied" });
              }}
            >
              <Copy className="h-3.5 w-3.5" />
              Copy Session ID
            </Button>
            <Button variant="outline" size="sm" onClick={handleResetSession}>
              Rotate Session ID
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
