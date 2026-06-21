/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#14161B",
          soft: "#1E2129",
        },
        paper: "#FFFFFF",
        surface: {
          DEFAULT: "#F5F6F8",
          2: "#EDEFF3",
        },
        line: "#E4E7EC",
        text: {
          DEFAULT: "#14161B",
          2: "#586072",
          3: "#8A92A3",
        },
        brand: {
          DEFAULT: "#5B4BE3",
          deep: "#3F32B5",
          soft: "#EEEBFB",
        },
        cat: {
          1: "#5B4BE3",
          2: "#0EA5A4",
          3: "#F59E0B",
          4: "#E0457B",
          5: "#3B82F6",
          6: "#10B981",
          7: "#64748B",
        },
      },
      fontFamily: {
        display: ["Space Grotesk", "system-ui", "sans-serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 6px 20px rgba(20, 22, 27, 0.05)",
        cardHover: "0 10px 28px rgba(20, 22, 27, 0.09)",
      },
      borderRadius: {
        card: "14px",
      },
    },
  },
  plugins: [],
};
