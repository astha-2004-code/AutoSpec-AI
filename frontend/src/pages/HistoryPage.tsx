import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, Clock, History, Search, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { formatSubsystem } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/Toast";
import { VerificationBadge } from "@/components/ui/VerificationBadge";

export function HistoryPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [filterText, setFilterText] = useState("");

  const { data: history = [], isLoading } = useQuery({
    queryKey: ["history", 100],
    queryFn: () => api.getHistory(100),
  });

  const clearMutation = useMutation({
    mutationFn: api.clearHistory,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["overview-stats"] });
      toast({ variant: "info", title: "Session search history cleared" });
    },
  });

  const deleteEntryMutation = useMutation({
    mutationFn: api.deleteHistoryEntry,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["overview-stats"] });
    },
  });

  const filtered = history.filter((item) =>
    filterText.trim()
      ? item.query.toLowerCase().includes(filterText.trim().toLowerCase()) ||
        (item.subsystem_filter || "").toLowerCase().includes(filterText.trim().toLowerCase())
      : true
  );

  return (
    <div className="mx-auto max-w-[1320px] space-y-6 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Session Search History
          </h1>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Audit trail of specification queries, latency telemetry, and verification statuses
            scoped to your workspace session.
          </p>
        </div>
        {history.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => clearMutation.mutate()}
            disabled={clearMutation.isPending}
          >
            <Trash2 className="h-3.5 w-3.5 text-red-500" />
            Clear All History
          </Button>
        )}
      </div>

      <div className="relative max-w-md">
        <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
        <input
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          placeholder="Filter past queries or subsystems..."
          className="w-full rounded-md border border-slate-300 bg-white py-2 pl-9 pr-3 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-slate-700 dark:bg-[#11161D] dark:text-slate-100"
        />
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 animate-pulse rounded-lg bg-slate-200 dark:bg-slate-800" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white py-14 text-center dark:border-slate-800 dark:bg-[#11161D]">
          <History className="mx-auto h-8 w-8 text-slate-400" />
          <p className="mt-3 text-sm font-medium text-slate-900 dark:text-white">
            No matching query history
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Run specification queries in Ask AutoSpec to build a reproducible session trail.
          </p>
          <Link to="/workspace" className="mt-4 inline-block">
            <Button size="sm">Open Ask AutoSpec</Button>
          </Link>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#11161D]">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-200 bg-slate-50 text-[11px] uppercase tracking-wider text-slate-500 dark:border-slate-800 dark:bg-[#0F141B]">
              <tr>
                <th className="px-4 py-3">Query</th>
                <th className="px-4 py-3">Subsystem</th>
                <th className="px-4 py-3">Verification</th>
                <th className="px-4 py-3">Hits / Latency</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {filtered.map((item) => (
                <tr key={item.id} className="hover:bg-slate-50/70 dark:hover:bg-slate-900/40">
                  <td className="max-w-md px-4 py-3 font-medium text-slate-900 dark:text-white">
                    {item.query}
                    <div className="mt-0.5 flex items-center gap-1 font-mono text-[10px] text-slate-400">
                      <Clock className="h-3 w-3" />
                      {item.created_at ? new Date(item.created_at).toLocaleString() : ""}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-slate-600 dark:text-slate-400">
                    {item.subsystem_filter ? formatSubsystem(item.subsystem_filter) : "Auto-Route"}
                  </td>
                  <td className="px-4 py-3">
                    {item.top_status ? <VerificationBadge status={item.top_status} /> : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-600 dark:text-slate-400">
                    {item.result_count} • {item.latency_ms} ms
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="inline-flex items-center gap-1.5">
                      <Link to={`/workspace?q=${encodeURIComponent(item.query)}`}>
                        <Button variant="outline" size="sm">
                          Re-run
                          <ArrowUpRight className="h-3.5 w-3.5" />
                        </Button>
                      </Link>
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="Delete history item"
                        onClick={() => deleteEntryMutation.mutate(item.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5 text-red-500" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
