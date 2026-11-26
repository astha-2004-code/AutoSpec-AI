import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ChevronRight,
  Cpu,
  FileStack,
  GitCompare,
  Github,
  History,
  LayoutDashboard,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Settings,
  Sun,
  TableProperties,
  X,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const SIDEBAR_ITEMS = [
  { to: "/overview", label: "Overview", icon: LayoutDashboard },
  { to: "/workspace", label: "Ask AutoSpec", icon: Search },
  { to: "/documents", label: "Document Library", icon: FileStack },
  { to: "/explorer", label: "Specification Explorer", icon: TableProperties },
  { to: "/compare", label: "Vehicle Comparison", icon: GitCompare },
  { to: "/history", label: "Search History", icon: History },
  { to: "/analytics", label: "Evaluation", icon: Activity },
  { to: "/architecture", label: "Architecture", icon: Cpu },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function AutoSpecLogo() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className="shrink-0"
    >
      <rect
        x="2"
        y="2"
        width="20"
        height="20"
        rx="4"
        className="stroke-accent-600 dark:stroke-accent-500"
        strokeWidth="1.75"
      />
      <path
        d="M7 16L12 6.5L17 16"
        className="stroke-slate-900 dark:stroke-white"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M9 13H15"
        className="stroke-accent-600 dark:stroke-accent-500"
        strokeWidth="1.75"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [dark, setDark] = useState<boolean>(() => {
    const saved = localStorage.getItem("autospec_theme");
    return saved ? saved === "dark" : true;
  });
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const location = useLocation();
  const isLanding = location.pathname === "/";

  const {
    data: health,
    isLoading: healthLoading,
    isError: healthError,
  } = useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
    retry: 1,
    staleTime: 60_000,
  });

  useEffect(() => {
    const root = document.documentElement;
    if (dark) {
      root.classList.add("dark");
      localStorage.setItem("autospec_theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("autospec_theme", "light");
    }
  }, [dark]);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  const activeItem = SIDEBAR_ITEMS.find((item) => item.to === location.pathname);

  const backendStatusBadge = (() => {
    if (healthLoading) {
      return {
        dot: "bg-amber-400 animate-pulse",
        label: "Waking API...",
      };
    }
    if (healthError || !health) {
      return {
        dot: "bg-red-500",
        label: "API Offline",
      };
    }
    return {
      dot: health.database_ready ? "bg-emerald-500" : "bg-amber-500",
      label: health.pgvector_enabled
        ? "Neon pgvector"
        : `${(health.database_backend || "sqlite").toUpperCase()} Ready`,
    };
  })();

  return (
    <div className="flex min-h-screen flex-col bg-[#F8FAFC] text-slate-900 dark:bg-[#0B0F14] dark:text-slate-100">
      {/* Top Navbar */}
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur dark:border-slate-800 dark:bg-[#11161D]/95">
        <div className="flex h-14 items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-3">
            {!isLanding && (
              <button
                onClick={() => setSidebarCollapsed((c) => !c)}
                aria-label="Toggle workspace sidebar"
                className="hidden h-8 w-8 items-center justify-center rounded-md border border-slate-200 text-slate-600 hover:bg-slate-50 lg:inline-flex dark:border-slate-800 dark:text-slate-400 dark:hover:bg-slate-800"
              >
                {sidebarCollapsed ? (
                  <PanelLeftOpen className="h-4 w-4" />
                ) : (
                  <PanelLeftClose className="h-4 w-4" />
                )}
              </button>
            )}

            <Link to="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
              <AutoSpecLogo />
              <span className="text-sm font-semibold text-slate-900 dark:text-white">
                AutoSpec <span className="text-accent-600 dark:text-accent-500">AI</span>
              </span>
            </Link>

            {isLanding ? (
              <nav className="ml-4 hidden items-center gap-1 md:flex" aria-label="Primary links">
                {[
                  { to: "/overview", label: "Workspace" },
                  { to: "/workspace", label: "Live Demo" },
                  { to: "/analytics", label: "Benchmarks" },
                  { to: "/architecture", label: "Architecture" },
                ].map((link) => (
                  <Link
                    key={link.to}
                    to={link.to}
                    className="rounded-md px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white"
                  >
                    {link.label}
                  </Link>
                ))}
              </nav>
            ) : (
              activeItem && (
                <div className="hidden items-center gap-1.5 text-xs text-slate-500 sm:flex">
                  <ChevronRight className="h-3.5 w-3.5" />
                  <span className="font-medium text-slate-900 dark:text-slate-200">
                    {activeItem.label}
                  </span>
                </div>
              )
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Live Backend & pgvector Telemetry Pill */}
            <Link
              to="/settings"
              className="hidden items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-100 sm:inline-flex dark:border-slate-800 dark:bg-[#0B0F14] dark:text-slate-300 dark:hover:bg-slate-800"
              title="Inspect database and vector runtime diagnostics"
            >
              <span className={cn("h-2 w-2 rounded-full", backendStatusBadge.dot)} />
              <span className="font-mono">{backendStatusBadge.label}</span>
            </Link>

            <a
              href="https://github.com/astha-2004-code/AutoSpec-AI"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub repository"
              className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-200 px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              <Github className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">GitHub</span>
            </a>

            <button
              onClick={() => setDark((d) => !d)}
              aria-label="Toggle color theme"
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>

            <button
              onClick={() => setMobileOpen((o) => !o)}
              aria-label="Toggle navigation menu"
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 text-slate-700 lg:hidden dark:border-slate-700 dark:text-slate-300"
            >
              {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </button>
          </div>
        </div>

        {/* Mobile Drawer */}
        {mobileOpen && (
          <nav className="border-t border-slate-200 bg-white px-4 py-2 lg:hidden dark:border-slate-800 dark:bg-[#11161D]">
            <div className="grid grid-cols-2 gap-1">
              {SIDEBAR_ITEMS.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium",
                      isActive
                        ? "bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-white"
                        : "text-slate-600 dark:text-slate-400"
                    )
                  }
                >
                  <Icon className="h-3.5 w-3.5" />
                  {label}
                </NavLink>
              ))}
            </div>
          </nav>
        )}
      </header>

      {/* Cold-Start / Degraded API Notice */}
      {!bannerDismissed && !isLanding && healthError && (
        <div className="flex items-center justify-between border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-800 dark:text-amber-300">
          <span>
            Backend API did not respond yet. Free-tier Render instances may require 30–50 seconds to
            cold-start.
          </span>
          <button
            onClick={() => setBannerDismissed(true)}
            className="text-[11px] font-medium underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Body with Collapsible Enterprise SaaS Sidebar for App Routes */}
      <div className="flex min-h-0 flex-1">
        {!isLanding && (
          <aside
            className={cn(
              "hidden shrink-0 flex-col border-r border-slate-200 bg-white transition-all duration-150 lg:flex dark:border-slate-800 dark:bg-[#11161D]",
              sidebarCollapsed ? "w-14" : "w-56"
            )}
          >
            <nav className="flex-1 space-y-1 p-2" aria-label="Workspace navigation">
              {SIDEBAR_ITEMS.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  title={label}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs font-medium transition-colors",
                      isActive
                        ? "bg-accent-600/10 text-accent-600 dark:text-accent-400"
                        : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/70 dark:hover:text-slate-100"
                    )
                  }
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {!sidebarCollapsed && <span className="truncate">{label}</span>}
                </NavLink>
              ))}
            </nav>

            {!sidebarCollapsed && (
              <div className="border-t border-slate-200 p-3 text-[11px] text-slate-500 dark:border-slate-800 dark:text-slate-400">
                <p className="font-semibold text-slate-700 dark:text-slate-300">AutoSpec AI v2.0</p>
                <p className="mt-0.5">Verify torque values against OEM publications.</p>
              </div>
            )}
          </aside>
        )}

        <main className="min-w-0 flex-1">{children}</main>
      </div>
    </div>
  );
}
