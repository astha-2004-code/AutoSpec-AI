import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "@/App";
import { VerificationBadge } from "@/components/ui/VerificationBadge";

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/v1/health")) {
        return new Response(
          JSON.stringify({
            status: "healthy",
            version: "2.0.0",
            environment: "test",
            storage_ready: true,
            database_ready: true,
            database_backend: "sqlite",
            pgvector_enabled: false,
            demo_ready: true,
            active_llm_provider: "none",
            embedding_model: "mock",
            reranker_enabled: false,
            ocr_available: false,
            document_count: 2,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/analytics/benchmark")) {
        return new Response(
          JSON.stringify({
            evaluated_at: "2026-11-26T00:00:00Z",
            configuration: {
              embedder: "DeterministicMockEmbedder(384)",
              top_k: 5,
              llm_provider: "none",
              documents_indexed: [],
              total_cases: 362,
              tuning_cases: 150,
              held_out_cases: 212,
            },
            system_performance: { ingestion_time_ms: 420, peak_traced_memory_mb: 12.4 },
            overall: {
              retrieval_mode: "hybrid_rerank",
              total_cases: 362,
              answerable_cases: 342,
              negative_cases: 20,
              metrics: {
                precision_at_k: 0.3039,
                recall_at_k: 0.8918,
                mrr: 0.8745,
                ndcg_at_k: 0.8619,
                exact_value_match_accuracy: 0.883,
                component_accuracy: 0.8889,
                unit_accuracy: 0.8947,
                citation_accuracy: 0.8977,
                evidence_grounding_rate: 0.8509,
                hallucination_rate: 0.0166,
                abstention_accuracy: 0.95,
                negative_rejection_rate: 0.95,
                p50_latency_ms: 6.17,
                p95_latency_ms: 9.74,
              },
              by_category: {},
              failed_queries: [],
            },
            ablation_study: {},
            limitations: "",
          }),
          { status: 200 }
        );
      }
      if (url.includes("/api/v1/query")) {
        return new Response(
          JSON.stringify({
            query: "Front caliper guide pin",
            results: [
              {
                document_id: "demo_apex_m400_2022",
                component: "Brake caliper guide pin bolts",
                spec_type: "torque",
                value: "37",
                unit: "Nm",
                alternate_value: "27",
                alternate_unit: "lb-ft",
                section: "SECTION 206-03: Front Disc Brake",
                subsystem: "front_brakes",
                page_number: 2,
                printed_page_label: "206-03-1",
                source_document: "2022_Apex_M400_Utility_Workshop_Manual.pdf",
                source_quote: "Brake caliper guide pin bolts | 37 | 27",
                evidence_bboxes: [],
                retrieval_score: 0.92,
                verification_status: "verified",
                extraction_method: "deterministic_fallback",
              },
            ],
            total_results: 1,
            extraction_method: "deterministic_fallback",
            retrieval_mode: "hybrid_rerank",
            processing_time_ms: 6.2,
            status: "answered",
            clarification_needed: false,
            active_context: {},
          }),
          { status: 200 }
        );
      }
      return new Response(JSON.stringify([]), { status: 200 });
    })
  );
});

describe("AutoSpec AI Frontend", () => {
  it("renders landing page headline and primary CTAs", async () => {
    render(<App />);
    expect(screen.getByText(/Engineering Specifications\./i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Launch Workspace/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Try Live Demo/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Upload Manual/i })).toBeInTheDocument();
  });

  it("renders verification status badges accurately", () => {
    render(
      <div>
        <VerificationBadge status="verified" />
        <VerificationBadge status="conflicting_evidence" />
        <VerificationBadge status="insufficient_evidence" />
      </div>
    );
    expect(screen.getByText("Verified")).toBeInTheDocument();
    expect(screen.getByText("Conflicting Evidence")).toBeInTheDocument();
    expect(screen.getByText("Insufficient Evidence")).toBeInTheDocument();
  });
});
