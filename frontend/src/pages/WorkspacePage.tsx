import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Check,
  Copy,
  Download,
  FileText,
  History,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  RotateCcw,
  Search,
  Send,
  Trash2,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatSubsystem } from "@/lib/utils";
import type { RetrievalMode, VehicleSpecification } from "@/types/api";
import { Button } from "@/components/ui/Button";
import { VerificationBadge } from "@/components/ui/VerificationBadge";
import { useToast } from "@/components/ui/Toast";
import { PdfEvidenceViewer, type ActiveCitation } from "@/components/PdfEvidenceViewer";

const EXAMPLE_QUERIES = [
  "What is the torque for front brake caliper guide pin bolts on 2022 Apex M400?",
  "What is the torque for rear brake caliper guide pin bolts on 2022 Apex M400?",
  "What is the cylinder head bolt torque on 2022 Apex M400?",
  "What about the tightening sequence?",
  "What is the brake disc minimum thickness for front disc brake on 2023 Apex S250?",
  "What is the engine oil capacity on 2022 Apex M400?",
];

export function WorkspacePage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();

  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [subsystemFilter, setSubsystemFilter] = useState<string>("");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>(
    () => (localStorage.getItem("autospec_pref_retrieval_mode") as RetrievalMode) || "hybrid_rerank"
  );
  const [queryText, setQueryText] = useState(() => searchParams.get("q") || "");
  const [conversationId] = useState(() => `conv_${Math.random().toString(36).slice(2, 10)}`);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<ActiveCitation | null>(null);

  const { data: documents = [], isLoading: docsLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: () => api.listDocuments(),
  });

  const { data: history = [] } = useQuery({
    queryKey: ["search-history"],
    queryFn: () => api.getHistory(20),
  });

  // Auto-select all authorized manuals initially
  useEffect(() => {
    if (documents.length > 0 && selectedDocIds.length === 0) {
      setSelectedDocIds(documents.map((d) => d.document_id));
    }
  }, [documents, selectedDocIds.length]);

  const availableSubsystems = useMemo(() => {
    const set = new Set<string>();
    documents.forEach((d) => d.subsystems.forEach((s) => set.add(s)));
    return Array.from(set).sort();
  }, [documents]);

  const queryMutation = useMutation({
    mutationFn: (q: string) =>
      api.querySpecs({
        query: q,
        top_k: 6,
        document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
        subsystem_filter: subsystemFilter || null,
        retrieval_mode: retrievalMode,
        conversation_id: conversationId,
      }),
    onSuccess: (resp) => {
      queryClient.invalidateQueries({ queryKey: ["search-history"] });
      if (resp.results.length > 0) {
        const first = resp.results[0];
        if (first.document_id) {
          setActiveCitation({
            documentId: first.document_id,
            documentName: first.source_document,
            vehicleLabel: [first.vehicle_year, first.vehicle_make, first.vehicle_model]
              .filter(Boolean)
              .join(" "),
            subsystem: first.subsystem,
            section: first.section,
            pageNumber: first.page_number,
            printedPageLabel: first.printed_page_label,
            sourceQuote: first.source_quote,
            verificationStatus: first.verification_status,
            evidenceBBoxes: first.evidence_bboxes || [],
          });
        }
      }
    },
    onError: (err) => {
      toast({
        variant: "error",
        title: "Query failed",
        description: err instanceof Error ? err.message : "Could not process query.",
      });
    },
  });

  const handleSubmit = (q?: string) => {
    const text = (q ?? queryText).trim();
    if (text.length < 2) return;
    if (q !== undefined) setQueryText(q);
    queryMutation.mutate(text);
  };

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const handleCopySpec = async (spec: VehicleSpecification, idx: number) => {
    const line = `${spec.component}: ${spec.value} ${spec.unit}${
      spec.alternate_value ? ` (${spec.alternate_value} ${spec.alternate_unit})` : ""
    } | ${spec.section} | Page ${spec.page_number} | ${spec.source_document}`;
    await navigator.clipboard.writeText(line);
    setCopiedId(`${idx}`);
    toast({ variant: "success", title: "Copied specification to clipboard" });
    setTimeout(() => setCopiedId(null), 1600);
  };

  const inspectCitation = (spec: VehicleSpecification) => {
    if (!spec.document_id) return;
    setActiveCitation({
      documentId: spec.document_id,
      documentName: spec.source_document,
      vehicleLabel: [spec.vehicle_year, spec.vehicle_make, spec.vehicle_model]
        .filter(Boolean)
        .join(" "),
      subsystem: spec.subsystem,
      section: spec.section,
      pageNumber: spec.page_number,
      printedPageLabel: spec.printed_page_label,
      sourceQuote: spec.source_quote,
      verificationStatus: spec.verification_status,
      evidenceBBoxes: spec.evidence_bboxes || [],
    });
    setRightOpen(true);
  };

  const handleResetThread = async () => {
    await api.resetConversation(conversationId);
    toast({ variant: "info", title: "Conversation context cleared" });
  };

  const handleClearHistory = async () => {
    await api.clearHistory();
    queryClient.invalidateQueries({ queryKey: ["search-history"] });
    toast({ variant: "info", title: "Search history cleared" });
  };

  const resp = queryMutation.data;

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col overflow-hidden lg:flex-row">
      {/* LEFT PANEL: Document Library, Subsystem Filters, Search History */}
      {leftOpen && (
        <aside className="flex w-full shrink-0 flex-col border-b border-slate-200 bg-white lg:w-72 lg:border-b-0 lg:border-r dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex items-center justify-between border-b border-slate-200 px-3.5 py-2.5 dark:border-slate-800">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Scope & Filters
            </span>
            <button
              onClick={() => setLeftOpen(false)}
              aria-label="Collapse left sidebar"
              className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800"
            >
              <PanelLeftClose className="h-4 w-4" />
            </button>
          </div>

          <div className="flex-1 space-y-5 overflow-y-auto p-3.5">
            {/* Vehicle / Manual Selection */}
            <div>
              <div className="mb-2 flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                  Authorized Manuals
                </label>
                <button
                  onClick={() => setSelectedDocIds(documents.map((d) => d.document_id))}
                  className="text-[11px] text-accent-600 hover:underline dark:text-accent-500"
                >
                  Select all
                </button>
              </div>
              {docsLoading ? (
                <div className="space-y-2">
                  <div className="h-10 animate-pulse rounded bg-slate-100 dark:bg-slate-800" />
                  <div className="h-10 animate-pulse rounded bg-slate-100 dark:bg-slate-800" />
                </div>
              ) : (
                <div className="space-y-1.5">
                  {documents.map((doc) => {
                    const checked = selectedDocIds.includes(doc.document_id);
                    return (
                      <label
                        key={doc.document_id}
                        className="flex cursor-pointer items-start gap-2 rounded border border-slate-200 p-2 text-xs hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/50"
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleDoc(doc.document_id)}
                          className="mt-0.5 rounded border-slate-300 text-accent-600 focus:ring-accent-500"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="truncate font-medium text-slate-900 dark:text-slate-100">
                            {doc.display_name}
                          </p>
                          <p className="text-[11px] text-slate-500 dark:text-slate-400">
                            {doc.total_pages} pp • {doc.spec_count} specs
                            {doc.is_demo ? " • Demo" : ""}
                          </p>
                        </div>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Subsystem Filter */}
            <div>
              <label
                htmlFor="subsystem-select"
                className="mb-1.5 block text-xs font-semibold text-slate-800 dark:text-slate-200"
              >
                Subsystem Filter
              </label>
              <select
                id="subsystem-select"
                value={subsystemFilter}
                onChange={(e) => setSubsystemFilter(e.target.value)}
                className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-100"
              >
                <option value="">All Subsystems (Auto-Route)</option>
                {availableSubsystems.map((sub) => (
                  <option key={sub} value={sub}>
                    {formatSubsystem(sub)}
                  </option>
                ))}
              </select>
            </div>

            {/* Retrieval Mode */}
            <div>
              <label
                htmlFor="mode-select"
                className="mb-1.5 block text-xs font-semibold text-slate-800 dark:text-slate-200"
              >
                Retrieval Strategy
              </label>
              <select
                id="mode-select"
                value={retrievalMode}
                onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}
                className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-100"
              >
                <option value="hybrid_rerank">Hybrid RRF + Rerank (Recommended)</option>
                <option value="hybrid">Hybrid RRF (Dense + BM25)</option>
                <option value="bm25">Sparse Lexical (BM25Plus)</option>
                <option value="dense">Dense Vector (FAISS)</option>
              </select>
            </div>

            {/* Search History */}
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-800 dark:text-slate-200">
                  <History className="h-3.5 w-3.5 text-slate-500" />
                  Recent Queries
                </span>
                {history.length > 0 && (
                  <button
                    onClick={handleClearHistory}
                    aria-label="Clear search history"
                    className="text-[11px] text-slate-500 hover:text-red-500"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
              {history.length === 0 ? (
                <p className="text-[11px] text-slate-500 dark:text-slate-500">
                  No queries in this session yet.
                </p>
              ) : (
                <div className="space-y-1">
                  {history.slice(0, 8).map((h) => (
                    <button
                      key={h.id}
                      onClick={() => handleSubmit(h.query)}
                      className="w-full rounded px-2 py-1.5 text-left text-xs text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800/70"
                    >
                      <p className="truncate font-medium text-slate-800 dark:text-slate-200">
                        {h.query}
                      </p>
                      <p className="font-mono text-[10px] text-slate-500">
                        {h.result_count} hits • {h.latency_ms} ms
                      </p>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </aside>
      )}

      {/* CENTER PANEL: Intelligent & Conversational Search + Structured Results */}
      <section className="flex min-w-0 flex-1 flex-col bg-[#F8FAFC] dark:bg-[#0B0F14]">
        {/* Workspace Toolbar */}
        <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2.5 dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex items-center gap-2">
            {!leftOpen && (
              <Button
                variant="outline"
                size="icon"
                aria-label="Open left sidebar"
                onClick={() => setLeftOpen(true)}
              >
                <PanelLeftOpen className="h-4 w-4" />
              </Button>
            )}
            <h1 className="text-sm font-semibold text-slate-900 dark:text-white">
              Specification Intelligence Workspace
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleResetThread}
              title="Reset conversational follow-up context"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">New Thread</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => api.downloadExport("csv", { document_ids: selectedDocIds })}
            >
              <Download className="h-3.5 w-3.5" />
              CSV
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => api.downloadExport("json", { document_ids: selectedDocIds })}
            >
              <Download className="h-3.5 w-3.5" />
              JSON
            </Button>
            {!rightOpen && (
              <Button
                variant="outline"
                size="icon"
                aria-label="Open PDF viewer"
                onClick={() => setRightOpen(true)}
              >
                <PanelRightOpen className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Search Input + Quick Chips */}
        <div className="border-b border-slate-200 bg-white px-4 py-3.5 dark:border-slate-800 dark:bg-[#0F141B]">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSubmit();
            }}
            className="flex gap-2"
          >
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="Ask a specification or follow-up question (e.g., 'What is the cylinder head bolt torque?' then 'What about the tightening sequence?')"
                aria-label="Specification query"
                className="h-9 w-full rounded-md border border-slate-300 bg-white pl-9 pr-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500 dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-100"
              />
            </div>
            <Button type="submit" disabled={queryMutation.isPending || queryText.trim().length < 2}>
              <Send className="h-3.5 w-3.5" />
              Search
            </Button>
          </form>

          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {EXAMPLE_QUERIES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => handleSubmit(ex)}
                className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-600 hover:border-accent-500 hover:text-accent-600 dark:border-slate-800 dark:bg-[#151C24] dark:text-slate-400 dark:hover:text-accent-400"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        {/* Results Pane */}
        <div className="flex-1 overflow-y-auto p-4">
          {queryMutation.isPending && (
            <div className="space-y-3">
              {[1, 2].map((i) => (
                <div
                  key={i}
                  className="rounded-md border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#11161D]"
                >
                  <div className="h-4 w-1/3 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
                  <div className="mt-3 h-7 w-1/4 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
                  <div className="mt-3 h-12 w-full animate-pulse rounded bg-slate-100 dark:bg-slate-800/60" />
                </div>
              ))}
            </div>
          )}

          {!queryMutation.isPending && !resp && (
            <div className="mx-auto mt-12 max-w-md text-center">
              <FileText className="mx-auto h-8 w-8 text-slate-400" />
              <h2 className="mt-3 text-sm font-semibold text-slate-900 dark:text-white">
                Ready for Evidence-Grounded Queries
              </h2>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                Select one or more manuals on the left, enter a technical query above, and inspect
                extracted specifications alongside original PDF page coordinates on the right.
              </p>
            </div>
          )}

          {!queryMutation.isPending && resp && (
            <div className="space-y-4">
              {/* Query Telemetry Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
                <span>
                  <strong className="text-slate-800 dark:text-slate-200">
                    {resp.total_results}
                  </strong>{" "}
                  result(s)
                  {resp.normalized_query && resp.normalized_query !== resp.query && (
                    <span className="ml-2 font-mono text-[11px]">
                      Normalized: "{resp.normalized_query}"
                    </span>
                  )}
                </span>
                <span className="font-mono text-[11px]">
                  {resp.processing_time_ms.toFixed(1)} ms • {resp.retrieval_mode} •{" "}
                  {resp.extraction_method}
                </span>
              </div>

              {/* Clarification Prompt */}
              {resp.clarification_needed && resp.clarification_prompt && (
                <div className="flex items-start gap-2.5 rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-800 dark:text-amber-300">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <div>
                    <p className="font-semibold">Clarification Recommended</p>
                    <p className="mt-0.5">{resp.clarification_prompt}</p>
                  </div>
                </div>
              )}

              {/* Abstention / Not Found State */}
              {resp.results.length === 0 && (
                <div className="rounded-md border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]">
                  <VerificationBadge
                    status={
                      resp.status === "insufficient_evidence"
                        ? "insufficient_evidence"
                        : "not_found"
                    }
                  />
                  <h3 className="mt-2.5 text-sm font-semibold text-slate-900 dark:text-white">
                    No Verified Specification Returned
                  </h3>
                  <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                    {resp.missing_info_explanation ||
                      "The selected manuals do not contain verifiable evidence matching this query."}
                  </p>
                </div>
              )}

              {/* Specification Result Cards */}
              {resp.results.map((spec, idx) => (
                <article
                  key={`${spec.document_id}_${spec.component}_${spec.page_number}_${idx}`}
                  className="rounded-md border border-slate-200 bg-white p-4 shadow-sm transition-colors hover:border-slate-300 dark:border-slate-800 dark:bg-[#11161D] dark:hover:border-slate-700"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                          {spec.component}
                        </h3>
                        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] uppercase text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                          {spec.spec_type}
                        </span>
                      </div>
                      <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                        {[spec.vehicle_year, spec.vehicle_make, spec.vehicle_model]
                          .filter(Boolean)
                          .join(" ")}{" "}
                        • {spec.section} •{" "}
                        <span className="font-mono">{formatSubsystem(spec.subsystem)}</span>
                      </p>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <VerificationBadge status={spec.verification_status} />
                      <span className="font-mono text-[10px] text-slate-400">
                        Retrieval Score: {spec.retrieval_score.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  {/* Primary & Alternate Values */}
                  <div className="mt-3 flex flex-wrap items-baseline gap-3">
                    <span className="font-mono text-2xl font-semibold text-accent-600 dark:text-accent-500">
                      {spec.value} {spec.unit}
                    </span>
                    {spec.alternate_value && (
                      <span className="font-mono text-sm text-slate-500 dark:text-slate-400">
                        / {spec.alternate_value} {spec.alternate_unit}
                      </span>
                    )}
                    {spec.conditions && (
                      <span className="rounded border border-slate-200 px-2 py-0.5 text-[11px] text-slate-600 dark:border-slate-700 dark:text-slate-300">
                        Condition: {spec.conditions}
                      </span>
                    )}
                    {spec.part_number && (
                      <span className="font-mono text-xs text-slate-600 dark:text-slate-300">
                        Part #: {spec.part_number}
                      </span>
                    )}
                  </div>

                  {/* Verbatim Quote */}
                  <blockquote className="mt-3 rounded border-l-2 border-accent-600 bg-slate-50 px-3 py-2 font-mono text-xs text-slate-700 dark:bg-[#0B0F14] dark:text-slate-300">
                    "{spec.source_quote}"
                  </blockquote>

                  {spec.verification_notes && (
                    <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
                      {spec.verification_notes}
                    </p>
                  )}

                  {/* Citation & Copy Actions */}
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2.5 dark:border-slate-800">
                    <button
                      type="button"
                      onClick={() => inspectCitation(spec)}
                      className="inline-flex items-center gap-1.5 text-xs font-medium text-accent-600 hover:underline dark:text-accent-500"
                    >
                      <FileText className="h-3.5 w-3.5" />
                      {spec.source_document} — Page {spec.page_number}
                      {spec.printed_page_label &&
                      spec.printed_page_label !== String(spec.page_number)
                        ? ` (Manual ${spec.printed_page_label})`
                        : ""}
                    </button>

                    <div className="flex items-center gap-1.5">
                      <Button variant="ghost" size="sm" onClick={() => handleCopySpec(spec, idx)}>
                        {copiedId === `${idx}` ? (
                          <Check className="h-3.5 w-3.5 text-emerald-500" />
                        ) : (
                          <Copy className="h-3.5 w-3.5" />
                        )}
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => inspectCitation(spec)}
                        disabled={!spec.document_id}
                      >
                        Inspect PDF
                      </Button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* RIGHT PANEL: Interactive PDF.js Evidence Viewer */}
      {rightOpen && (
        <aside className="flex h-96 w-full shrink-0 flex-col border-t border-slate-200 lg:h-auto lg:w-[440px] xl:w-[500px] lg:border-l lg:border-t-0 dark:border-slate-800">
          <div className="flex items-center justify-between border-b border-slate-200 bg-white px-3.5 py-2.5 dark:border-slate-800 dark:bg-[#11161D]">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Source PDF Evidence
            </span>
            <button
              onClick={() => setRightOpen(false)}
              aria-label="Collapse PDF panel"
              className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800"
            >
              <PanelRightClose className="h-4 w-4" />
            </button>
          </div>
          <div className="min-h-0 flex-1">
            <PdfEvidenceViewer citation={activeCitation} />
          </div>
        </aside>
      )}
    </div>
  );
}
