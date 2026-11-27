import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// Mock pdfjs-dist in jsdom unit tests so canvas worker isn't needed
vi.mock("pdfjs-dist", () => ({
  GlobalWorkerOptions: { workerSrc: "" },
  getDocument: vi.fn(),
}));

vi.mock("pdfjs-dist/build/pdf.worker.min.mjs?url", () => ({
  default: "mock-worker.js",
}));
