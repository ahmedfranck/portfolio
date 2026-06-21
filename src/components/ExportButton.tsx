import { Download } from "lucide-react";
import { exportToCsv } from "../lib/csv";

interface ExportButtonProps {
  filename: string;
  rows: Record<string, unknown>[];
  label?: string;
}

export default function ExportButton({ filename, rows, label = "Exporter en CSV" }: ExportButtonProps) {
  return (
    <button
      type="button"
      onClick={() => exportToCsv(filename, rows)}
      disabled={rows.length === 0}
      className="inline-flex items-center gap-2 rounded-full border border-brand bg-brand text-white px-4 py-2 text-sm font-medium transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
    >
      <Download size={16} aria-hidden="true" />
      {label}
    </button>
  );
}
