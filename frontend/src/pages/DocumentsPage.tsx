import { useEffect, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Eye,
  FileText,
  Lock,
  Pencil,
  RefreshCw,
  RotateCcw,
  Trash2,
  Upload,
  X,
  XCircle,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatBytes, formatSubsystem } from "@/lib/utils";
import type { DocumentInfo, IngestionJobStatus } from "@/types/api";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/Toast";
import { PdfEvidenceViewer, type ActiveCitation } from "@/components/PdfEvidenceViewer";

const PIPELINE_STAGES = [
  "uploading",
  "parsing",
  "chunking",
  "embedding",
  "indexing",
  "validating",
  "completed",
];

const PAGE_SIZE = 6;

export function DocumentsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [dragActive, setDragActive] = useState(false);
  const [manufacturer, setManufacturer] = useState("");
  const [model, setModel] = useState("");
  const [year, setYear] = useState("");
  const [enableOcr, setEnableOcr] = useState(false);
  const [activeJob, setActiveJob] = useState<IngestionJobStatus | null>(null);
  const [editingDoc, setEditingDoc] = useState<DocumentInfo | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [previewDoc, setPreviewDoc] = useState<DocumentInfo | null>(null);
  const [page, setPage] = useState(1);

  const { data: documents = [], isLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: () => api.listDocuments(),
  });

  // Poll active background ingestion job for real stage + page-by-page progress
  useEffect(() => {
    if (!activeJob || !["queued", "running"].includes(activeJob.status)) return;

    const timer = setInterval(async () => {
      try {
        const updated = await api.getJobStatus(activeJob.job_id);
        setActiveJob(updated);
        if (updated.status === "completed") {
          queryClient.invalidateQueries({ queryKey: ["documents"] });
          queryClient.invalidateQueries({ queryKey: ["overview-stats"] });
          toast({
            variant: "success",
            title: updated.is_duplicate ? "Duplicate manual linked" : "Manual indexed successfully",
            description: updated.message,
          });
        } else if (updated.status === "failed") {
          toast({
            variant: "error",
            title: "Ingestion failed",
            description: updated.error || updated.message,
          });
        } else if (updated.status === "cancelled") {
          toast({ variant: "info", title: "Ingestion cancelled" });
        }
      } catch {
        // ignore transient poll errors
      }
    }, 600);

    return () => clearInterval(timer);
  }, [activeJob, queryClient, toast]);

  const uploadMutation = useMutation({
    mutationFn: (file: File) =>
      api.uploadDocument(file, {
        manufacturer: manufacturer.trim() || undefined,
        model: model.trim() || undefined,
        year: year ? Number(year) : undefined,
        enable_ocr: enableOcr,
      }),
    onSuccess: (job) => {
      setActiveJob(job);
      if (job.is_duplicate) {
        queryClient.invalidateQueries({ queryKey: ["documents"] });
        toast({
          variant: "info",
          title: "Duplicate file detected",
          description: job.message,
        });
      } else {
        toast({
          variant: "info",
          title: "Upload accepted",
          description: `Processing ${job.filename}...`,
        });
      }
    },
    onError: (err) => {
      toast({
        variant: "error",
        title: "Upload rejected",
        description: err instanceof Error ? err.message : "Invalid PDF upload.",
      });
    },
  });

  const retryMutation = useMutation({
    mutationFn: (jobId: string) => api.retryJob(jobId),
    onSuccess: (job) => {
      setActiveJob(job);
      toast({ variant: "info", title: "Retrying ingestion job" });
    },
    onError: (err) => {
      toast({
        variant: "error",
        title: "Retry failed",
        description: err instanceof Error ? err.message : "Could not retry ingestion job.",
      });
    },
  });

  const handleFileSelected = (file: File | undefined) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast({
        variant: "error",
        title: "Invalid file format",
        description: "Please upload a .pdf manual.",
      });
      return;
    }
    uploadMutation.mutate(file);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    handleFileSelected(file);
  };

  const deleteMutation = useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["overview-stats"] });
      toast({ variant: "success", title: "Document deleted" });
    },
    onError: (err) => {
      toast({
        variant: "error",
        title: "Delete failed",
        description: err instanceof Error ? err.message : "Cannot delete document.",
      });
    },
  });

  const reindexMutation = useMutation({
    mutationFn: (docId: string) => api.reindexDocument(docId, enableOcr),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast({ variant: "success", title: "Reindex complete", description: res.message });
    },
    onError: (err) => {
      toast({
        variant: "error",
        title: "Reindex failed",
        description: err instanceof Error ? err.message : "Failed to reindex manual.",
      });
    },
  });

  const renameMutation = useMutation({
    mutationFn: ({ id, display_name }: { id: string; display_name: string }) =>
      api.renameDocument(id, { display_name }),
    onSuccess: () => {
      setEditingDoc(null);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast({ variant: "success", title: "Document renamed" });
    },
    onError: (err) => {
      toast({
        variant: "error",
        title: "Rename failed",
        description: err instanceof Error ? err.message : "Could not rename document.",
      });
    },
  });

  const totalPages = Math.max(1, Math.ceil(documents.length / PAGE_SIZE));
  const paginatedDocs = documents.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const previewCitation: ActiveCitation | null = previewDoc
    ? {
        documentId: previewDoc.document_id,
        documentName: previewDoc.filename,
        vehicleLabel:
          `${previewDoc.year || ""} ${previewDoc.manufacturer} ${previewDoc.model}`.trim(),
        subsystem: previewDoc.subsystems[0] || "general",
        section: previewDoc.display_name,
        pageNumber: 1,
        printedPageLabel: "1",
        sourceQuote: "",
        verificationStatus: "verified",
        evidenceBBoxes: [],
      }
    : null;

  return (
    <div className="mx-auto max-w-[1320px] space-y-8 px-4 py-8 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Document Library & Ingestion
          </h1>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Upload OEM service manuals into your isolated session workspace. Public demonstration
            manuals remain read-only.
          </p>
        </div>
        <Link to="/workspace">
          <Button variant="outline">Open Ask AutoSpec</Button>
        </Link>
      </div>

      {/* Drag & Drop PDF Upload */}
      <div className="grid gap-6 lg:grid-cols-12">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center transition-colors lg:col-span-7 ${
            dragActive
              ? "border-accent-500 bg-accent-500/5"
              : "border-slate-300 bg-white dark:border-slate-700 dark:bg-[#11161D]"
          }`}
        >
          <Upload className="h-7 w-7 text-accent-600 dark:text-accent-500" />
          <p className="mt-3 text-sm font-medium text-slate-900 dark:text-white">
            Drag and drop an automotive workshop manual PDF here
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Validated against PDF signature, page bounds, and size limits. Duplicate manuals are
            linked automatically.
          </p>
          <label className="mt-4 cursor-pointer">
            <input
              type="file"
              accept=".pdf,application/pdf"
              className="sr-only"
              onChange={(e) => handleFileSelected(e.target.files?.[0])}
            />
            <span className="inline-flex h-9 items-center rounded-md bg-accent-600 px-4 text-sm font-medium text-white hover:bg-accent-700">
              Select PDF File
            </span>
          </label>
        </div>

        {/* Optional Vehicle Metadata Tags */}
        <div className="rounded-lg border border-slate-200 bg-white p-5 lg:col-span-5 dark:border-slate-800 dark:bg-[#11161D]">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            Upload Vehicle Metadata (Optional)
          </h2>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            Auto-detected from manual cover pages when left blank.
          </p>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400">
                Manufacturer
              </label>
              <input
                value={manufacturer}
                onChange={(e) => setManufacturer(e.target.value)}
                placeholder="e.g. Ford, Toyota"
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400">Model</label>
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="e.g. F-150, Camry"
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400">Year</label>
              <input
                type="number"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder="e.g. 2022"
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
              />
            </div>
            <div className="flex items-end pb-1">
              <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-700 dark:text-slate-300">
                <input
                  type="checkbox"
                  checked={enableOcr}
                  onChange={(e) => setEnableOcr(e.target.checked)}
                />
                Enable OCR fallback
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Real-Time Stage & Page Ingestion Progress Bar */}
      {activeJob && (
        <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <p className="text-sm font-semibold text-slate-900 dark:text-white">
                  {activeJob.filename} — <span className="uppercase">{activeJob.status}</span>
                </p>
                {activeJob.is_duplicate && (
                  <span className="inline-flex items-center gap-1 rounded bg-sky-500/10 px-2 py-0.5 text-[10px] font-medium text-sky-600 dark:text-sky-400">
                    <CheckCircle2 className="h-3 w-3" /> Duplicate Linked
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400">
                {activeJob.message} ({activeJob.current_page}/{activeJob.total_pages || "?"} pages)
              </p>
            </div>
            <div className="flex items-center gap-2">
              {["queued", "running"].includes(activeJob.status) && (
                <Button variant="outline" size="sm" onClick={() => api.cancelJob(activeJob.job_id)}>
                  <XCircle className="h-3.5 w-3.5" />
                  Cancel
                </Button>
              )}
              {["failed", "cancelled"].includes(activeJob.status) && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => retryMutation.mutate(activeJob.job_id)}
                  disabled={retryMutation.isPending}
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Retry
                </Button>
              )}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-1.5 text-[10px]">
            {PIPELINE_STAGES.map((st) => {
              const currentIdx = PIPELINE_STAGES.indexOf(activeJob.stage);
              const idx = PIPELINE_STAGES.indexOf(st);
              const active = st === activeJob.stage;
              const done = currentIdx > idx || activeJob.status === "completed";
              return (
                <span
                  key={st}
                  className={`rounded px-2 py-0.5 font-mono uppercase ${
                    done
                      ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"
                      : active
                        ? "bg-accent-600/15 font-semibold text-accent-600 dark:text-accent-400"
                        : "bg-slate-100 text-slate-400 dark:bg-slate-800"
                  }`}
                >
                  {st}
                </span>
              );
            })}
          </div>

          <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
            <div
              className="h-full bg-accent-600 transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(4, activeJob.progress_pct))}%` }}
            />
          </div>
        </div>
      )}

      {/* Indexed Collection */}
      <div>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            Indexed Document Collections ({documents.length})
          </h2>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 text-xs">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Prev
              </Button>
              <span className="font-mono text-slate-500">
                {page} / {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                Next
              </Button>
            </div>
          )}
        </div>

        {isLoading ? (
          <div className="grid gap-4 md:grid-cols-2">
            {[1, 2].map((i) => (
              <div
                key={i}
                className="h-36 animate-pulse rounded-lg bg-slate-200 dark:bg-slate-800"
              />
            ))}
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {paginatedDocs.map((doc) => (
              <div
                key={doc.document_id}
                className="flex flex-col justify-between rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#11161D]"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-start gap-2.5">
                      <FileText className="mt-0.5 h-5 w-5 shrink-0 text-accent-600 dark:text-accent-500" />
                      <div>
                        {editingDoc?.document_id === doc.document_id ? (
                          <form
                            onSubmit={(e) => {
                              e.preventDefault();
                              renameMutation.mutate({
                                id: doc.document_id,
                                display_name: editTitle,
                              });
                            }}
                            className="flex gap-1.5"
                          >
                            <input
                              value={editTitle}
                              onChange={(e) => setEditTitle(e.target.value)}
                              className="rounded border border-slate-300 px-2 py-1 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
                            />
                            <Button size="sm" type="submit">
                              Save
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => setEditingDoc(null)}>
                              Cancel
                            </Button>
                          </form>
                        ) : (
                          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                            {doc.display_name}
                          </h3>
                        )}
                        <p className="mt-0.5 font-mono text-[11px] text-slate-500">
                          {doc.year || ""} {doc.manufacturer} {doc.model} • {doc.filename}
                        </p>
                      </div>
                    </div>

                    {doc.read_only && (
                      <span
                        title="Public demonstration manuals cannot be modified or deleted"
                        className="inline-flex items-center gap-1 rounded border border-slate-200 bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
                      >
                        <Lock className="h-3 w-3" /> Read-Only Demo
                      </span>
                    )}
                  </div>

                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {doc.subsystems.slice(0, 6).map((sub) => (
                      <span
                        key={sub}
                        className="rounded bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                      >
                        {formatSubsystem(sub)}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-3 text-xs text-slate-500 dark:border-slate-800">
                  <span className="font-mono text-[11px]">
                    {doc.total_pages} pp • {doc.spec_count} specs •{" "}
                    {formatBytes(doc.file_size_bytes)}
                  </span>

                  <div className="flex items-center gap-1.5">
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Preview PDF manual"
                      onClick={() => setPreviewDoc(doc)}
                    >
                      <Eye className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={doc.read_only}
                      onClick={() => {
                        setEditingDoc(doc);
                        setEditTitle(doc.display_name);
                      }}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={doc.read_only || reindexMutation.isPending}
                      onClick={() => reindexMutation.mutate(doc.document_id)}
                      title="Reindex manual"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={doc.read_only || deleteMutation.isPending}
                      onClick={() => deleteMutation.mutate(doc.document_id)}
                      title="Delete manual"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-red-500" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* PDF Preview Modal */}
      {previewDoc && previewCitation && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="flex h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#11161D]">
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
              <div>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                  {previewDoc.display_name}
                </h3>
                <p className="font-mono text-[11px] text-slate-500">{previewDoc.filename}</p>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setPreviewDoc(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="min-h-0 flex-1">
              <PdfEvidenceViewer citation={previewCitation} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
