/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        surface: {
          50: "#F8FAFC",
          100: "#F1F5F9",
          800: "#151C24",
          900: "#11161D",
          950: "#0B0F14",
        },
        accent: {
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1D4ED8",
        },
      },
    },
  },
  plugins: [],
};

