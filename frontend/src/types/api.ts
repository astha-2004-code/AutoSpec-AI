export type VerificationStatus =
  | "verified"
  | "partially_supported"
  | "conflicting_evidence"
  | "ambiguous"
  | "insufficient_evidence"
  | "not_found"
  | "unsupported"
  | "unverified";

export type RetrievalMode = "hybrid_rerank" | "hybrid" | "dense" | "bm25";

export interface EvidenceBBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  page_width: number;
  page_height: number;
}

export interface VehicleSpecification {
  document_id?: string | null;
  component: string;
  spec_type: string;
  value: string;
  unit: string;
  alternate_value?: string | null;
  alternate_unit?: string | null;
  normalized_value?: number | null;
  normalized_unit?: string | null;
  range_min?: string | null;
  range_max?: string | null;
  tolerance?: string | null;
  conditions?: string | null;
  part_number?: string | null;
  material?: string | null;
  section: string;
  subsystem?: string | null;
  page_number: number;
  printed_page_label?: string | null;
  source_document: string;
  vehicle_make?: string | null;
  vehicle_model?: string | null;
  vehicle_year?: number | null;
  source_quote: string;
  evidence_bboxes: EvidenceBBox[];
  retrieval_score: number;
  verification_status: VerificationStatus | string;
  verification_notes?: string | null;
  extraction_method: string;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  display_name: string;
  fingerprint: string;
  total_pages: number;
  indexed_chunks: number;
  spec_count: number;
  freshness: string;
  created_at: string;
  is_demo: boolean;
  read_only: boolean;
  manufacturer: string;
  model: string;
  year?: number | null;
  subsystems: string[];
  ocr_pages: number;
  file_size_bytes: number;
}

export interface IngestionJobStatus {
  job_id: string;
  document_id?: string | null;
  filename: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  stage: string;
  current_page: number;
  total_pages: number;
  progress_pct: number;
  is_duplicate?: boolean;
  message: string;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface QueryRequestPayload {
  query: string;
  top_k?: number;
  document_ids?: string[];
  subsystem_filter?: string | null;
  manufacturer_filter?: string | null;
  model_filter?: string | null;
  year_filter?: number | null;
  retrieval_mode?: RetrievalMode;
  conversation_id?: string | null;
}

export interface QueryResponse {
  query: string;
  normalized_query?: string | null;
  results: VehicleSpecification[];
  total_results: number;
  extraction_method: string;
  retrieval_mode: RetrievalMode;
  processing_time_ms: number;
  status: string;
  clarification_needed: boolean;
  clarification_prompt?: string | null;
  missing_info_explanation?: string | null;
  conversation_id?: string | null;
  active_context: Record<string, unknown>;
  message?: string | null;
}

export interface SearchHistoryEntry {
  id: number;
  query: string;
  document_ids: string[];
  subsystem_filter?: string | null;
  result_count: number;
  top_status?: string | null;
  latency_ms: number;
  created_at: string;
}

export interface ComparisonCell {
  document_id: string;
  vehicle_label: string;
  status: "present" | "missing" | "incomparable";
  original_value?: string | null;
  original_unit?: string | null;
  normalized_value?: number | null;
  normalized_unit?: string | null;
  alternate_value?: string | null;
  alternate_unit?: string | null;
  page_number?: number | null;
  printed_page_label?: string | null;
  section?: string | null;
  source_quote?: string | null;
  verification_status?: string | null;
  evidence_bboxes: EvidenceBBox[];
}

export interface ComparisonRow {
  canonical_component: string;
  spec_type: string;
  subsystem: string;
  cells: ComparisonCell[];
  has_variance: boolean;
  comparable: boolean;
  notes?: string | null;
}

export interface ComparisonResponse {
  vehicles: DocumentInfo[];
  rows: ComparisonRow[];
  total_rows: number;
  subsystems_covered: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
  environment?: string;
  storage_ready: boolean;
  database_ready: boolean;
  database_backend?: string;
  pgvector_enabled?: boolean;
  demo_ready: boolean;
  active_llm_provider: string;
  embedding_model: string;
  reranker_enabled: boolean;
  ocr_available: boolean;
  document_count: number;
}

export interface OverviewStats {
  document_count: number;
  demo_document_count: number;
  user_document_count: number;
  total_pages: number;
  total_chunks: number;
  total_specifications: number;
  verified_specifications: number;
  query_count: number;
  subsystems: string[];
}

export interface DemoWorkspaceResponse {
  demo_ready: boolean;
  documents: DocumentInfo[];
  sample_queries: string[];
}

export interface EvaluationRunSummary {
  run_id: string;
  mode: string;
  dataset_version: string;
  total_cases: number;
  top_1_accuracy: number;
  top_3_recall: number;
  mrr: number;
  exact_numeric_accuracy: number;
  unit_accuracy: number;
  citation_accuracy: number;
  avg_latency_ms: number;
  ablation: Record<string, unknown>;
  created_at: string;
}

export interface BenchmarkMetrics {
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  exact_value_match_accuracy: number;
  component_accuracy: number;
  unit_accuracy: number;
  citation_accuracy: number;
  evidence_grounding_rate: number;
  hallucination_rate: number;
  abstention_accuracy: number;
  negative_rejection_rate: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

export interface BenchmarkReport {
  evaluated_at: string;
  configuration: {
    embedder: string;
    top_k: number;
    llm_provider: string;
    documents_indexed: string[];
    total_cases: number;
    tuning_cases: number;
    held_out_cases: number;
  };
  system_performance: {
    ingestion_time_ms: number;
    peak_traced_memory_mb: number;
  };
  overall: {
    retrieval_mode: string;
    total_cases: number;
    answerable_cases: number;
    negative_cases: number;
    metrics: BenchmarkMetrics;
    by_category: Record<string, { total: number; passed: number; accuracy: number }>;
    failed_queries: Array<{
      id: string;
      category: string;
      query: string;
      expected: string;
      actual: string;
      reason: string;
    }>;
  };
  tuning_split?: BenchmarkMetrics | null;
  held_out_split?: BenchmarkMetrics | null;
  ablation_study: Record<string, BenchmarkMetrics>;
  limitations: string;
}
