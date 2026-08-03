import { useState } from "react";
import { ExternalLink } from "lucide-react";
import type { ConsultingReport } from "../config/content";

export default function PowerBIEmbed({ report }: { report: ConsultingReport }) {
  const [loaded, setLoaded] = useState(false);

  return (
    <article className="flex h-full flex-col rounded-card border border-line bg-white p-4 shadow-sm transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover">
      <h3 className="font-display text-base font-semibold text-ink">{report.title}</h3>
      <div className="relative mt-3 aspect-[16/10] w-full overflow-hidden rounded-card border border-line bg-surface">
        {!loaded && (
          <div className="absolute inset-0 z-10 overflow-hidden bg-surface" role="status" aria-live="polite">
            <div className="h-9 animate-pulse border-b border-line bg-surface-2" />
            <div className="grid h-[calc(100%-2.25rem)] animate-pulse grid-cols-3 gap-3 p-4">
              <span className="rounded bg-surface-2" />
              <span className="col-span-2 rounded bg-surface-2" />
            </div>
            <span className="sr-only">Chargement du rapport {report.title}</span>
          </div>
        )}
        <iframe
          src={report.src}
          title={report.title}
          loading="lazy"
          allow="fullscreen"
          allowFullScreen
          onLoad={() => setLoaded(true)}
          className={`absolute inset-0 h-full w-full border-0 transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
        />
      </div>
      <a
        href={report.src}
        target="_blank"
        rel="noreferrer"
        className="group mt-3 inline-flex w-fit items-center gap-1.5 text-sm font-medium text-brand transition-colors duration-200 hover:text-brand-deep"
      >
        Ouvrir en plein écran
        <ExternalLink size={14} aria-hidden="true" className="transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
      </a>
    </article>
  );
}
