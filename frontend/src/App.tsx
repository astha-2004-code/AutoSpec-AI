import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/Layout";
import { ToastProvider } from "@/components/ui/Toast";
import { AnalyticsPage } from "@/pages/AnalyticsPage";
import { ArchitecturePage } from "@/pages/ArchitecturePage";
import { ComparePage } from "@/pages/ComparePage";
import { DocumentsPage } from "@/pages/DocumentsPage";
import { ExplorerPage } from "@/pages/ExplorerPage";
import { HistoryPage } from "@/pages/HistoryPage";
import { LandingPage } from "@/pages/LandingPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { WorkspacePage } from "@/pages/WorkspacePage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <BrowserRouter>
          <AppShell>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/overview" element={<OverviewPage />} />
              <Route path="/workspace" element={<WorkspacePage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/explorer" element={<ExplorerPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/architecture" element={<ArchitecturePage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </AppShell>
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
}

export default App;
