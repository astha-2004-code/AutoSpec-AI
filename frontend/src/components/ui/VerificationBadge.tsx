import type { ReactNode } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  XCircle,
  ShieldAlert,
  SearchX,
} from "lucide-react";
import { cn } from "@/lib/utils";

const STATUS_MAP: Record<
  string,
  { label: string; className: string; icon: ReactNode; tooltip: string }
> = {
  verified: {
    label: "Verified",
    className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
    icon: <CheckCircle2 className="h-3.5 w-3.5" />,
    tooltip: "Value, unit, component, subsystem, and verbatim quote confirmed on source page.",
  },
  partially_supported: {
    label: "Partially Supported",
    className: "border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-400",
    icon: <AlertTriangle className="h-3.5 w-3.5" />,
    tooltip:
      "Value and quote confirmed on page, but unit or condition required table-header context.",
  },
  conflicting_evidence: {
    label: "Conflicting Evidence",
    className: "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-400",
    icon: <ShieldAlert className="h-3.5 w-3.5" />,
    tooltip: "Incompatible measurements or conflicting values detected across retrieved evidence.",
  },
  ambiguous: {
    label: "Ambiguous",
    className: "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400",
    icon: <HelpCircle className="h-3.5 w-3.5" />,
    tooltip: "Multiple subsystems or variants match; specify assembly (e.g. front vs rear).",
  },
  insufficient_evidence: {
    label: "Insufficient Evidence",
    className: "border-slate-400/40 bg-slate-500/10 text-slate-700 dark:text-slate-300",
    icon: <SearchX className="h-3.5 w-3.5" />,
    tooltip: "Source evidence lacks complete provenance required for verification.",
  },
  not_found: {
    label: "Not Found",
    className: "border-slate-400/40 bg-slate-500/10 text-slate-600 dark:text-slate-400",
    icon: <SearchX className="h-3.5 w-3.5" />,
    tooltip: "No matching specification found in authorized manuals.",
  },
  unsupported: {
    label: "Unsupported",
    className: "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400",
    icon: <XCircle className="h-3.5 w-3.5" />,
    tooltip: "Value or unit could not be grounded in the cited page text.",
  },
};

export function VerificationBadge({ status }: { status: string }) {
  const cfg = STATUS_MAP[status] || STATUS_MAP.insufficient_evidence;
  return (
    <span
      title={cfg.tooltip}
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium",
        cfg.className
      )}
    >
      {cfg.icon}
      {cfg.label}
    </span>
  );
}
