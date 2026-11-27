import { useEffect, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import {
  ChevronLeft,
  ChevronRight,
  FileText,
  Maximize2,
  Minimize2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { getDocumentPdfUrl, getSessionId } from "@/lib/api";
import type { EvidenceBBox } from "@/types/api";
import { Button } from "@/components/ui/Button";
import { VerificationBadge } from "@/components/ui/VerificationBadge";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

export interface ActiveCitation {
  documentId: string;
  documentName: string;
  vehicleLabel?: string;
  subsystem?: string | null;
  section?: string;
  pageNumber: number;
  printedPageLabel?: string | null;
  sourceQuote: string;
  verificationStatus?: string;
  evidenceBBoxes: EvidenceBBox[];
}

interface PdfEvidenceViewerProps {
  citation: ActiveCitation | null;
}

export function PdfEvidenceViewer({ citation }: PdfEvidenceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [pdfDoc, setPdfDoc] = useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [scale, setScale] = useState<number>(1.1);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState<boolean>(false);
  const [viewportSize, setViewportSize] = useState<{ width: number; height: number }>({
    width: 612,
    height: 792,
  });

  // Load PDF document whenever citation.documentId changes
  const targetDocId = citation?.documentId;
  const targetPage = citation?.pageNumber ?? 1;

  useEffect(() => {
    if (!targetDocId) {
      setPdfDoc(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    const url = getDocumentPdfUrl(targetDocId);
    const loadingTask = pdfjsLib.getDocument({
      url,
      httpHeaders: { "X-Session-ID": getSessionId() },
    });

    loadingTask.promise
      .then((doc) => {
        if (cancelled) {
          doc.destroy();
          return;
        }
        setPdfDoc(doc);
        setNumPages(doc.numPages);
        setCurrentPage(Math.min(Math.max(1, targetPage), doc.numPages));
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load source PDF document.");
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetDocId]);

  // Navigate to cited page whenever pageNumber changes on the same doc
  useEffect(() => {
    if (citation?.pageNumber && numPages > 0) {
      setCurrentPage(Math.min(Math.max(1, citation.pageNumber), numPages));
    }
  }, [citation?.pageNumber, numPages]);

  // Render active page to canvas
  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;

    let cancelled = false;
    let renderTask: pdfjsLib.RenderTask | null = null;

    pdfDoc.getPage(currentPage).then((page) => {
      if (cancelled || !canvasRef.current) return;
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      setViewportSize({ width: viewport.width, height: viewport.height });

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      renderTask = page.render({ canvasContext: ctx, viewport });
      renderTask.promise.catch(() => {
        // Ignore cancelled renders on rapid navigation
      });
    });

    return () => {
      cancelled = true;
      if (renderTask) renderTask.cancel();
    };
  }, [pdfDoc, currentPage, scale]);

  if (!citation) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center">
        <div className="mb-3 rounded-lg border border-slate-200 bg-slate-100 p-3 dark:border-slate-800 dark:bg-slate-900">
          <FileText className="h-6 w-6 text-slate-500 dark:text-slate-400" />
        </div>
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Interactive PDF Evidence Viewer
        </h3>
        <p className="mt-1.5 max-w-xs text-xs leading-relaxed text-slate-600 dark:text-slate-400">
          Select any extracted specification or citation badge to inspect the original OEM workshop
          manual page and coordinate-grounded highlight.
        </p>
      </div>
    );
  }

  const showHighlightOverlay =
    currentPage === citation.pageNumber && citation.evidenceBBoxes.length > 0;
  const printedDiffers =
    citation.printedPageLabel && citation.printedPageLabel !== String(citation.pageNumber);

  const viewerBody = (
    <div className="flex h-full flex-col bg-slate-50 dark:bg-[#0D1218]">
      {/* Evidence Metadata Header */}
      <div className="border-b border-slate-200 bg-white px-3.5 py-2.5 dark:border-slate-800 dark:bg-[#11161D]">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
              {citation.documentName}
            </p>
            <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-slate-500 dark:text-slate-400">
              {citation.vehicleLabel && <span>{citation.vehicleLabel}</span>}
              {citation.subsystem && (
                <span className="font-mono text-accent-600 dark:text-accent-500">
                  {citation.subsystem}
                </span>
              )}
              <span>
                PDF Page{" "}
                <strong className="text-slate-700 dark:text-slate-200">{currentPage}</strong>/
                {numPages || "?"}
              </span>
              {printedDiffers && currentPage === citation.pageNumber && (
                <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  Manual Label: {citation.printedPageLabel}
                </span>
              )}
            </div>
          </div>
          {citation.verificationStatus && (
            <VerificationBadge status={citation.verificationStatus} />
          )}
        </div>

        {/* Toolbar */}
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2 dark:border-slate-800/80">
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              aria-label="Previous page"
              disabled={currentPage <= 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="px-2 font-mono text-xs text-slate-600 dark:text-slate-300">
              {currentPage} / {numPages || 1}
            </span>
            <Button
              variant="outline"
              size="icon"
              aria-label="Next page"
              disabled={currentPage >= numPages}
              onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            {currentPage !== citation.pageNumber && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCurrentPage(citation.pageNumber)}
                className="ml-1 text-xs text-accent-600 dark:text-accent-500"
              >
                Jump to Cited p.{citation.pageNumber}
              </Button>
            )}
          </div>

          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              aria-label="Zoom out"
              onClick={() => setScale((s) => Math.max(0.6, Number((s - 0.15).toFixed(2))))}
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </Button>
            <span className="w-11 text-center font-mono text-xs text-slate-600 dark:text-slate-300">
              {Math.round(scale * 100)}%
            </span>
            <Button
              variant="outline"
              size="icon"
              aria-label="Zoom in"
              onClick={() => setScale((s) => Math.min(2.4, Number((s + 0.15).toFixed(2))))}
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              aria-label={fullscreen ? "Exit full screen" : "Full screen"}
              onClick={() => setFullscreen((f) => !f)}
            >
              {fullscreen ? (
                <Minimize2 className="h-3.5 w-3.5" />
              ) : (
                <Maximize2 className="h-3.5 w-3.5" />
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Original Evidence Quote Strip */}
      <div className="border-b border-slate-200 bg-slate-100/80 px-3.5 py-2 dark:border-slate-800 dark:bg-[#0F151D]">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Grounded Source Quote
        </p>
        <p className="mt-0.5 font-mono text-xs text-slate-800 dark:text-slate-200">
          "{citation.sourceQuote}"
        </p>
      </div>

      {/* PDF Page Canvas + Real Coordinate Highlight Layer */}
      <div className="relative flex-1 overflow-auto p-4">
        {loading && (
          <div className="flex h-64 flex-col items-center justify-center gap-2">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent-600 border-t-transparent" />
            <p className="text-xs text-slate-500 dark:text-slate-400">Rendering PDF page...</p>
          </div>
        )}

        {error && (
          <div className="mx-auto mt-8 max-w-sm rounded-md border border-red-500/30 bg-red-500/5 p-4 text-center">
            <p className="text-xs font-medium text-red-600 dark:text-red-400">{error}</p>
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={() => setCurrentPage((p) => p)}
            >
              <RotateCcw className="h-3.5 w-3.5" /> Retry
            </Button>
          </div>
        )}

        {!loading && !error && (
          <div
            className="relative mx-auto shadow-md ring-1 ring-slate-200 dark:ring-slate-800"
            style={{ width: viewportSize.width, height: viewportSize.height }}
          >
            <canvas ref={canvasRef} className="block bg-white" />
            {showHighlightOverlay &&
              citation.evidenceBBoxes.map((box, idx) => {
                const leftPct = (box.x0 / (box.page_width || 612)) * 100;
                const topPct = (box.y0 / (box.page_height || 792)) * 100;
                const widthPct = ((box.x1 - box.x0) / (box.page_width || 612)) * 100;
                const heightPct = ((box.y1 - box.y0) / (box.page_height || 792)) * 100;

                return (
                  <div
                    key={idx}
                    aria-label="Highlighted specification evidence"
                    className="pointer-events-none absolute rounded-sm border-2 border-accent-600 bg-accent-500/20"
                    style={{
                      left: `${leftPct}%`,
                      top: `${topPct}%`,
                      width: `${Math.max(widthPct, 2)}%`,
                      height: `${Math.max(heightPct, 1.5)}%`,
                    }}
                  />
                );
              })}
          </div>
        )}
      </div>
    </div>
  );

  if (fullscreen) {
    return (
      <div className="fixed inset-0 z-50 bg-black/75 p-4 backdrop-blur-sm">
        <div className="mx-auto h-full max-w-6xl overflow-hidden rounded-lg border border-slate-700">
          {viewerBody}
        </div>
      </div>
    );
  }

  return viewerBody;
}
