import { Cpu, Database, Lock, ShieldCheck } from "lucide-react";

export function ArchitecturePage() {
  return (
    <div className="mx-auto max-w-[1120px] space-y-8 px-4 py-8 sm:px-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
          System & RAG Architecture
        </h1>
        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
          Production design choices prioritizing verifiable grounding, multi-document isolation, and
          deterministic CPU fallback.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {[
          {
            icon: Database,
            title: "1. Layout-Aware Ingestion & Multi-Document Isolation",
            items: [
              "PyMuPDF table finder + multi-page header continuation + line-level PDF coordinate capture.",
              "SHA-256 document fingerprinting with atomic FAISS, JSON BM25, and page-evidence persistence.",
              "Distinguishes physical PDF page index from printed workshop manual page labels.",
              "SQLAlchemy repository (SQLite dev / PostgreSQL prod) isolating session uploads from read-only demo manuals.",
            ],
          },
          {
            icon: Cpu,
            title: "2. Hybrid RRF Retrieval & Second-Stage Reranking",
            items: [
              "Query normalization expanding automotive abbreviations (ft-lb, ATF, CCA, diff, specs) and detecting vehicle year/model.",
              "Reciprocal Rank Fusion (k=60) combining dense SentenceTransformer embeddings and automotive-tokenized BM25Plus.",
              "Cross-assembly disambiguation isolating front vs. rear brakes and 4WD vs. RWD subsystems.",
              "Lightweight CPU reranker (optional CrossEncoder + deterministic domain feature scorer).",
            ],
          },
          {
            icon: ShieldCheck,
            title: "3. Strict Evidence Verification Pipeline",
            items: [
              "Boundary-aware numeric and range/tolerance validation (prevents partial-digit false positives like '3' in '37').",
              "Physical unit compatibility validation across torque, volume, length, pressure, and electrical dimensions.",
              "Verbatim quote alignment verified directly against persisted source page text.",
              "Six explicit verification states including Partially Supported, Conflicting Evidence, Ambiguous, and Not Found.",
            ],
          },
          {
            icon: Lock,
            title: "4. Security, Privacy & Concurrency",
            items: [
              "Strict PDF magic-byte, encryption, page-limit, and filename sanitization controls.",
              "Zero pickle deserialization — BM25 index persisted purely as safe JSON.",
              "Prompt-injection defense wrapping untrusted manual text and stripping override directives.",
              "Bounded background ThreadPoolExecutor keeping FastAPI's async event loop responsive with rate limiting and security headers.",
            ],
          },
        ].map(({ icon: Icon, title, items }) => (
          <div
            key={title}
            className="rounded-lg border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-[#11161D]"
          >
            <div className="flex items-center gap-2.5">
              <Icon className="h-5 w-5 text-accent-600 dark:text-accent-500" />
              <h2 className="text-sm font-semibold text-slate-900 dark:text-white">{title}</h2>
            </div>
            <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
              {items.map((it) => (
                <li key={it} className="flex gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-600" />
                  <span>{it}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
