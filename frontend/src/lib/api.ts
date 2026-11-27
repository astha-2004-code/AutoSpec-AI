import type {
  BenchmarkReport,
  ComparisonResponse,
  DemoWorkspaceResponse,
  DocumentInfo,
  EvaluationRunSummary,
  HealthResponse,
  IngestionJobStatus,
  OverviewStats,
  QueryRequestPayload,
  QueryResponse,
  SearchHistoryEntry,
  VehicleSpecification,
} from "@/types/api";

const RAW_API_BASE = (import.meta.env.VITE_API_BASE_URL || "").trim().replace(/\/$/, "");
const API_BASE = RAW_API_BASE || (import.meta.env.PROD ? "" : "http://localhost:8000");

const SESSION_STORAGE_KEY = "autospec_session_id";
const DEFAULT_TIMEOUT_MS = 35_000;

export function getApiBaseUrl(): string {
  return API_BASE || "(same-origin proxy)";
}

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!id) {
    id = `sess_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
    localStorage.setItem(SESSION_STORAGE_KEY, id);
  }
  return id;
}

export function resetSessionId(): string {
  const id = `sess_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
  localStorage.setItem(SESSION_STORAGE_KEY, id);
  return id;
}

async function apiFetch<T>(
  path: string,
  init?: RequestInit & { timeoutMs?: number; retries?: number }
): Promise<T> {
  const timeoutMs = init?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRetries = init?.retries ?? 1;
  const headers = new Headers(init?.headers || {});
  headers.set("X-Session-ID", getSessionId());
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if ((res.status === 502 || res.status === 503) && attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 1500));
        continue;
      }

      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const payload = await res.json();
          detail = payload.detail || detail;
        } catch {
          // ignore non-json error
        }
        throw new Error(detail);
      }
      return (await res.json()) as Promise<T>;
    } catch (err) {
      clearTimeout(timer);
      const e = err instanceof Error ? err : new Error("Network error");
      lastError =
        e.name === "AbortError"
          ? new Error(
              "Request timed out — Render free-tier instances may take 30-50s to wake on cold start."
            )
          : e;
      if (attempt < maxRetries && !(init?.body instanceof FormData)) {
        await new Promise((r) => setTimeout(r, 1200));
        continue;
      }
      throw lastError;
    }
  }
  throw lastError || new Error("Request failed");
}

export function getDocumentPdfUrl(documentId: string): string {
  return `${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/pdf`;
}

export const api = {
  getHealth: () => apiFetch<HealthResponse>("/api/v1/health", { timeoutMs: 25_000, retries: 1 }),

  getOverview: () => apiFetch<OverviewStats>("/api/v1/overview"),

  getDemo: () => apiFetch<DemoWorkspaceResponse>("/api/v1/demo"),

  listDocuments: (params?: {
    manufacturer?: string;
    model?: string;
    year?: number;
    search?: string;
  }) => {
    const qs = new URLSearchParams();
    if (params?.manufacturer) qs.set("manufacturer", params.manufacturer);
    if (params?.model) qs.set("model", params.model);
    if (params?.year) qs.set("year", String(params.year));
    if (params?.search) qs.set("search", params.search);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return apiFetch<DocumentInfo[]>(`/api/v1/documents${suffix}`);
  },

  uploadDocument: (
    file: File,
    meta?: { manufacturer?: string; model?: string; year?: number; enable_ocr?: boolean }
  ) => {
    const fd = new FormData();
    fd.append("file", file);
    if (meta?.manufacturer) fd.append("manufacturer", meta.manufacturer);
    if (meta?.model) fd.append("model", meta.model);
    if (meta?.year) fd.append("year", String(meta.year));
    if (meta?.enable_ocr) fd.append("enable_ocr", "true");
    return apiFetch<IngestionJobStatus>("/api/v1/documents/upload", {
      method: "POST",
      body: fd,
      timeoutMs: 90_000,
      retries: 0,
    });
  },

  getJobStatus: (jobId: string) =>
    apiFetch<IngestionJobStatus>(`/api/v1/jobs/${encodeURIComponent(jobId)}`),

  cancelJob: (jobId: string) =>
    apiFetch<IngestionJobStatus>(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    }),

  retryJob: (jobId: string) =>
    apiFetch<IngestionJobStatus>(`/api/v1/jobs/${encodeURIComponent(jobId)}/retry`, {
      method: "POST",
    }),

  renameDocument: (
    documentId: string,
    payload: { display_name?: string; manufacturer?: string; model?: string; year?: number }
  ) =>
    apiFetch<DocumentInfo>(`/api/v1/documents/${encodeURIComponent(documentId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  deleteDocument: (documentId: string) =>
    apiFetch<{ status: string }>(`/api/v1/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
    }),

  reindexDocument: (documentId: string, enableOcr = false) =>
    apiFetch<{ status: string; message: string }>(
      `/api/v1/documents/${encodeURIComponent(documentId)}/reindex?enable_ocr=${enableOcr}`,
      { method: "POST" }
    ),

  querySpecs: (payload: QueryRequestPayload) =>
    apiFetch<QueryResponse>("/api/v1/query", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  resetConversation: (conversationId: string) =>
    apiFetch<{ status: string }>(
      `/api/v1/conversations/${encodeURIComponent(conversationId)}/reset`,
      {
        method: "POST",
      }
    ),

  listSpecifications: (params?: {
    document_ids?: string[];
    subsystem?: string;
    spec_type?: string;
    verification_status?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.document_ids?.length) qs.set("document_ids", params.document_ids.join(","));
    if (params?.subsystem) qs.set("subsystem", params.subsystem);
    if (params?.spec_type) qs.set("spec_type", params.spec_type);
    if (params?.verification_status) qs.set("verification_status", params.verification_status);
    if (params?.search) qs.set("search", params.search);
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return apiFetch<VehicleSpecification[]>(`/api/v1/specifications${suffix}`);
  },

  compareVehicles: (payload: {
    document_ids: string[];
    subsystem?: string;
    spec_type?: string;
    component_query?: string;
    preferred_torque_unit?: string;
    preferred_volume_unit?: string;
    preferred_length_unit?: string;
  }) =>
    apiFetch<ComparisonResponse>("/api/v1/compare", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getHistory: (limit = 30) => apiFetch<SearchHistoryEntry[]>(`/api/v1/history?limit=${limit}`),

  clearHistory: () => apiFetch<{ deleted: number }>("/api/v1/history", { method: "DELETE" }),

  deleteHistoryEntry: (entryId: number) =>
    apiFetch<{ deleted: number }>(`/api/v1/history/${entryId}`, { method: "DELETE" }),

  getBenchmark: () => apiFetch<BenchmarkReport>("/api/v1/analytics/benchmark"),

  getEvaluationRuns: () => apiFetch<EvaluationRunSummary[]>("/api/v1/evaluation/runs"),

  downloadExport: async (
    format: "csv" | "json",
    params?: { document_ids?: string[]; subsystem?: string; spec_type?: string }
  ) => {
    const qs = new URLSearchParams();
    if (params?.document_ids?.length) qs.set("document_ids", params.document_ids.join(","));
    if (params?.subsystem) qs.set("subsystem", params.subsystem);
    if (params?.spec_type) qs.set("spec_type", params.spec_type);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    const res = await fetch(`${API_BASE}/api/v1/export/${format}${suffix}`, {
      headers: { "X-Session-ID": getSessionId() },
    });
    if (!res.ok) throw new Error("Export failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autospec_specifications.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  },

  downloadComparisonExport: async (
    format: "csv" | "json",
    payload: {
      document_ids: string[];
      subsystem?: string;
      spec_type?: string;
      component_query?: string;
      preferred_torque_unit?: string;
      preferred_volume_unit?: string;
      preferred_length_unit?: string;
    }
  ) => {
    const res = await fetch(`${API_BASE}/api/v1/compare/export?format=${format}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-ID": getSessionId(),
      },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Comparison export failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autospec_comparison.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  },
};
