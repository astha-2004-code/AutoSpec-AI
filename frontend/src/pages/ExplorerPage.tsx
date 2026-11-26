import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Copy, Download, FileText, Search, X } from "lucide-react";
import { api } from "@/lib/api";
import { formatSubsystem } from "@/lib/utils";
import type { VehicleSpecification } from "@/types/api";
import { Button } from "@/components/ui/Button";
import { VerificationBadge } from "@/components/ui/VerificationBadge";
import { useToast } from "@/components/ui/Toast";
import { PdfEvidenceViewer, type ActiveCitation } from "@/components/PdfEvidenceViewer";

export function ExplorerPage() {
  const { toast } = useToast();
  const [selectedDocId, setSelectedDocId] = useState("");
  const [subsystem, setSubsystem] = useState("");
  const [specType, setSpecType] = useState("");
  const [search, setSearch] = useState("");
  const [drawerCitation, setDrawerCitation] = useState<ActiveCitation | null>(null);

  const { data: documents = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: () => api.listDocuments(),
  });

  const { data: specs = [], isLoading } = useQuery({
    queryKey: ["all-specs", selectedDocId, subsystem, specType, search],
    queryFn: () =>
      api.listSpecifications({
        document_ids: selectedDocId ? [selectedDocId] : undefined,
        subsystem: subsystem || undefined,
        spec_type: specType || undefined,
        search: search.trim() || undefined,
        limit: 500,
      }),
  });

  const subsystems = useMemo(
    () => Array.from(new Set(specs.map((s) => s.subsystem).filter(Boolean) as string[])).sort(),
    [specs]
  );

  const specTypes = useMemo(
    () => Array.from(new Set(specs.map((s) => s.spec_type))).sort(),
    [specs]
  );

  const openEvidence = (spec: VehicleSpecification) => {
    if (!spec.document_id) return;
    setDrawerCitation({
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
  };

  return (
    <div className="mx-auto max-w-[1440px] space-y-5 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Specification Explorer
          </h1>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Browse, filter, verify, and export structured specifications across indexed workshop
            manuals.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              api.downloadExport("csv", {
                document_ids: selectedDocId ? [selectedDocId] : undefined,
                subsystem: subsystem || undefined,
                spec_type: specType || undefined,
              })
            }
          >
            <Download className="h-3.5 w-3.5" /> Export CSV
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              api.downloadExport("json", {
                document_ids: selectedDocId ? [selectedDocId] : undefined,
                subsystem: subsystem || undefined,
                spec_type: specType || undefined,
              })
            }
          >
            <Download className="h-3.5 w-3.5" /> Export JSON
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="grid gap-3 rounded-lg border border-slate-200 bg-white p-4 sm:grid-cols-2 lg:grid-cols-4 dark:border-slate-800 dark:bg-[#11161D]">
        <select
          value={selectedDocId}
          onChange={(e) => setSelectedDocId(e.target.value)}
          aria-label="Filter by manual"
          className="h-9 rounded border border-slate-300 bg-white px-2.5 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
        >
          <option value="">All Authorized Manuals</option>
          {documents.map((d) => (
            <option key={d.document_id} value={d.document_id}>
              {d.display_name}
            </option>
          ))}
        </select>

        <select
          value={subsystem}
          onChange={(e) => setSubsystem(e.target.value)}
          aria-label="Filter by subsystem"
          className="h-9 rounded border border-slate-300 bg-white px-2.5 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
        >
          <option value="">All Subsystems</option>
          {subsystems.map((sub) => (
            <option key={sub} value={sub}>
              {formatSubsystem(sub)}
            </option>
          ))}
        </select>

        <select
          value={specType}
          onChange={(e) => setSpecType(e.target.value)}
          aria-label="Filter by specification type"
          className="h-9 rounded border border-slate-300 bg-white px-2.5 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
        >
          <option value="">All Specification Categories</option>
          {specTypes.map((st) => (
            <option key={st} value={st}>
              {st}
            </option>
          ))}
        </select>

        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-slate-400" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search component, value, part #..."
            className="h-9 w-full rounded border border-slate-300 bg-white pl-8 pr-3 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#11161D]">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:border-slate-800 dark:bg-[#0F151D] dark:text-slate-400">
                <th className="px-4 py-3">Component</th>
                <th className="px-4 py-3">Value</th>
                <th className="px-4 py-3">Alternate</th>
                <th className="px-4 py-3">Subsystem</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Source Citation</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {isLoading && (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-slate-500">
                    Loading indexed specifications...
                  </td>
                </tr>
              )}
              {!isLoading && specs.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-slate-500">
                    No specifications match the selected filters.
                  </td>
                </tr>
              )}
              {specs.map((s, idx) => (
                <tr
                  key={`${s.document_id}_${s.component}_${s.page_number}_${idx}`}
                  className="hover:bg-slate-50 dark:hover:bg-slate-800/40"
                >
                  <td className="px-4 py-2.5 font-medium text-slate-900 dark:text-slate-100">
                    <p>{s.component}</p>
                    <p className="text-[11px] font-normal text-slate-500">
                      {[s.vehicle_year, s.vehicle_make, s.vehicle_model].filter(Boolean).join(" ")}
                    </p>
                  </td>
                  <td className="px-4 py-2.5 font-mono font-semibold text-accent-600 dark:text-accent-500">
                    {s.value} {s.unit}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-slate-500">
                    {s.alternate_value ? `${s.alternate_value} ${s.alternate_unit}` : "—"}
                  </td>
                  <td className="px-4 py-2.5 text-slate-600 dark:text-slate-300">
                    {formatSubsystem(s.subsystem)}
                  </td>
                  <td className="px-4 py-2.5">
                    <VerificationBadge status={s.verification_status} />
                  </td>
                  <td className="px-4 py-2.5">
                    <button
                      type="button"
                      onClick={() => openEvidence(s)}
                      className="inline-flex items-center gap-1 text-accent-600 hover:underline dark:text-accent-500"
                    >
                      <FileText className="h-3.5 w-3.5" />
                      p.{s.page_number}
                    </button>
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <button
                      type="button"
                      aria-label="Copy specification"
                      onClick={async () => {
                        await navigator.clipboard.writeText(`${s.component}: ${s.value} ${s.unit}`);
                        toast({ variant: "success", title: "Copied to clipboard" });
                      }}
                      className="rounded p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Slide-over PDF Evidence Drawer */}
      {drawerCitation && (
        <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Verified PDF Evidence
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setDrawerCitation(null)}
              aria-label="Close evidence drawer"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="min-h-0 flex-1">
            <PdfEvidenceViewer citation={drawerCitation} />
          </div>
        </div>
      )}
    </div>
  );
}
