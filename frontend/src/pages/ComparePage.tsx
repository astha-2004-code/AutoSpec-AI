import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, FileText, GitCompare, X } from "lucide-react";
import { api } from "@/lib/api";
import { formatSubsystem } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { PdfEvidenceViewer, type ActiveCitation } from "@/components/PdfEvidenceViewer";

export function ComparePage() {
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [subsystem, setSubsystem] = useState("");
  const [specType, setSpecType] = useState("");
  const [torqueUnit, setTorqueUnit] = useState("Nm");
  const [volumeUnit, setVolumeUnit] = useState("ml");
  const [lengthUnit, setLengthUnit] = useState("mm");
  const [componentSearch, setComponentSearch] = useState("");
  const [drawerCitation, setDrawerCitation] = useState<ActiveCitation | null>(null);

  const { data: documents = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: () => api.listDocuments(),
  });

  useEffect(() => {
    if (documents.length >= 2 && selectedDocs.length === 0) {
      setSelectedDocs(documents.slice(0, 2).map((d) => d.document_id));
    }
  }, [documents, selectedDocs.length]);

  const payload = {
    document_ids: selectedDocs,
    subsystem: subsystem || undefined,
    spec_type: specType || undefined,
    component_query: componentSearch.trim() || undefined,
    preferred_torque_unit: torqueUnit,
    preferred_volume_unit: volumeUnit,
    preferred_length_unit: lengthUnit,
  };

  const {
    data: comparison,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["compare-vehicles", payload],
    queryFn: () => api.compareVehicles(payload),
    enabled: selectedDocs.length >= 2,
  });

  const toggleVehicle = (id: string) => {
    setSelectedDocs((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  return (
    <div className="mx-auto max-w-[1440px] space-y-6 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Vehicle Specification Comparison
          </h1>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Normalize compatible engineering units across platforms while preserving verbatim
            original values and page citations.
          </p>
        </div>

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={selectedDocs.length < 2}
            onClick={() => api.downloadComparisonExport("csv", payload)}
          >
            <Download className="h-3.5 w-3.5" /> CSV Report
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={selectedDocs.length < 2}
            onClick={() => api.downloadComparisonExport("json", payload)}
          >
            <Download className="h-3.5 w-3.5" /> JSON Report
          </Button>
        </div>
      </div>

      {/* Controls */}
      <div className="grid gap-4 rounded-lg border border-slate-200 bg-white p-4 lg:grid-cols-12 dark:border-slate-800 dark:bg-[#11161D]">
        <div className="lg:col-span-5">
          <p className="mb-2 text-xs font-semibold text-slate-800 dark:text-slate-200">
            Select Vehicles (2–6)
          </p>
          <div className="flex flex-wrap gap-2">
            {documents.map((d) => {
              const active = selectedDocs.includes(d.document_id);
              return (
                <button
                  key={d.document_id}
                  type="button"
                  onClick={() => toggleVehicle(d.document_id)}
                  className={`rounded border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    active
                      ? "border-accent-600 bg-accent-600/10 text-accent-600 dark:text-accent-400"
                      : "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-400"
                  }`}
                >
                  {d.year || ""} {d.manufacturer} {d.model}
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5 lg:col-span-7">
          <div>
            <label className="mb-1 block text-[11px] text-slate-500">Subsystem</label>
            <select
              value={subsystem}
              onChange={(e) => setSubsystem(e.target.value)}
              className="h-8 w-full rounded border border-slate-300 bg-white px-2 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
            >
              <option value="">All</option>
              {(comparison?.subsystems_covered || []).map((s) => (
                <option key={s} value={s}>
                  {formatSubsystem(s)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-500">Category</label>
            <select
              value={specType}
              onChange={(e) => setSpecType(e.target.value)}
              className="h-8 w-full rounded border border-slate-300 bg-white px-2 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
            >
              <option value="">All</option>
              <option value="torque">Torque</option>
              <option value="fluid_capacity">Fluid Capacity</option>
              <option value="dimension_clearance">Dimensions</option>
              <option value="electrical">Electrical</option>
              <option value="pressure">Pressure</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-500">Torque Unit</label>
            <select
              value={torqueUnit}
              onChange={(e) => setTorqueUnit(e.target.value)}
              className="h-8 w-full rounded border border-slate-300 bg-white px-2 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
            >
              <option value="Nm">Nm</option>
              <option value="lb-ft">lb-ft</option>
              <option value="lb-in">lb-in</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-500">Volume Unit</label>
            <select
              value={volumeUnit}
              onChange={(e) => setVolumeUnit(e.target.value)}
              className="h-8 w-full rounded border border-slate-300 bg-white px-2 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
            >
              <option value="ml">ml</option>
              <option value="L">L</option>
              <option value="qt">qt</option>
              <option value="pt">pt</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-500">Filter Component</label>
            <input
              value={componentSearch}
              onChange={(e) => setComponentSearch(e.target.value)}
              placeholder="e.g. caliper"
              className="h-8 w-full rounded border border-slate-300 bg-white px-2 text-xs dark:border-slate-700 dark:bg-[#0B0F14]"
            />
          </div>
          <div className="col-span-2 sm:col-span-5 hidden">
            <select value={lengthUnit} onChange={(e) => setLengthUnit(e.target.value)}>
              <option value="mm">mm</option>
              <option value="in">in</option>
            </select>
          </div>
        </div>
      </div>

      {selectedDocs.length < 2 && (
        <div className="rounded-lg border border-slate-200 bg-white p-10 text-center dark:border-slate-800 dark:bg-[#11161D]">
          <GitCompare className="mx-auto h-7 w-7 text-slate-400" />
          <p className="mt-2 text-sm font-medium text-slate-700 dark:text-slate-300">
            Select at least two vehicles above to generate a side-by-side specification comparison.
          </p>
        </div>
      )}

      {error && (
        <p className="rounded border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-600 dark:text-red-400">
          {error instanceof Error ? error.message : "Comparison failed"}
        </p>
      )}

      {isLoading && selectedDocs.length >= 2 && (
        <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-xs text-slate-500 dark:border-slate-800 dark:bg-[#11161D]">
          Comparing specifications across selected vehicles...
        </div>
      )}

      {comparison && (
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#11161D]">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:border-slate-800 dark:bg-[#0F151D] dark:text-slate-400">
                  <th className="px-4 py-3">Component</th>
                  <th className="px-4 py-3">Subsystem</th>
                  {comparison.vehicles.map((v) => (
                    <th key={v.document_id} className="px-4 py-3">
                      {v.year || ""} {v.manufacturer} {v.model}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {comparison.rows.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="px-4 py-3 font-medium text-slate-900 dark:text-slate-100">
                      <p>{row.canonical_component}</p>
                      {row.has_variance && row.comparable && (
                        <span className="mt-0.5 inline-block rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-600 dark:text-amber-400">
                          Differs across models
                        </span>
                      )}
                      {!row.comparable && (
                        <span className="mt-0.5 inline-block rounded bg-rose-500/10 px-1.5 py-0.5 text-[10px] text-rose-600 dark:text-rose-400">
                          Incomparable units
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-500">{formatSubsystem(row.subsystem)}</td>
                    {row.cells.map((cell) => (
                      <td key={cell.document_id} className="px-4 py-3 align-top">
                        {cell.status === "missing" ? (
                          <span className="font-mono text-[11px] text-slate-400">Not listed</span>
                        ) : (
                          <div>
                            <p className="font-mono text-sm font-semibold text-slate-900 dark:text-white">
                              {cell.normalized_value !== null && cell.normalized_value !== undefined
                                ? `${cell.normalized_value} ${cell.normalized_unit}`
                                : `${cell.original_value} ${cell.original_unit || ""}`}
                            </p>
                            <p className="text-[11px] text-slate-500">
                              Source: {cell.original_value} {cell.original_unit || ""}
                            </p>
                            {cell.page_number && (
                              <button
                                type="button"
                                onClick={() =>
                                  setDrawerCitation({
                                    documentId: cell.document_id,
                                    documentName: cell.vehicle_label,
                                    vehicleLabel: cell.vehicle_label,
                                    subsystem: row.subsystem,
                                    section: cell.section || undefined,
                                    pageNumber: cell.page_number!,
                                    printedPageLabel: cell.printed_page_label,
                                    sourceQuote: cell.source_quote || "",
                                    verificationStatus: cell.verification_status || undefined,
                                    evidenceBBoxes: cell.evidence_bboxes || [],
                                  })
                                }
                                className="mt-1 inline-flex items-center gap-1 text-[11px] text-accent-600 hover:underline dark:text-accent-500"
                              >
                                <FileText className="h-3 w-3" /> p.{cell.page_number}
                              </button>
                            )}
                          </div>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {drawerCitation && (
        <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-[#11161D]">
          <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Comparison Source Evidence
            </span>
            <Button variant="ghost" size="icon" onClick={() => setDrawerCitation(null)}>
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
