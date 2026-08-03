import { useState } from "react";
import { BarChart3, ShieldCheck } from "lucide-react";
import { CONSULTING_CLIENTS, CONSULTING_DISCLAIMER, type ConsultingClient } from "../config/content";
import { getEmployerLogo } from "../lib/employerLogos";
import PowerBIEmbed from "./PowerBIEmbed";
import Reveal, { RevealGroup, RevealItem } from "./Reveal";

function monogram(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0])
    .join("")
    .toUpperCase();
}

function ConsultingClientLogo({ client }: { client: ConsultingClient }) {
  const src = getEmployerLogo(client.logo);
  const [failed, setFailed] = useState(false);
  const cropWhitespace = client.logo.endsWith(".png");

  if (!src || failed) {
    return (
      <span
        className="flex h-14 w-36 shrink-0 items-center justify-center rounded-lg border border-line bg-brand-soft font-display text-lg font-semibold text-brand"
        aria-hidden="true"
      >
        {monogram(client.name)}
      </span>
    );
  }

  return (
    <span className="flex h-14 w-48 max-w-full shrink-0 items-center justify-center overflow-hidden rounded-lg border border-line bg-white">
      <img
        src={src}
        alt={`Logo ${client.name}`}
        onError={() => setFailed(true)}
        className={`h-full w-full ${cropWhitespace ? "object-cover" : "object-contain p-2.5"}`}
      />
    </span>
  );
}

export default function ConsultingPortfolio() {
  return (
    <div className="mt-8">
      <Reveal className="flex gap-3 border-l-4 border-brand bg-brand-soft px-4 py-4 sm:px-5">
        <ShieldCheck size={22} aria-hidden="true" className="mt-0.5 shrink-0 text-brand" />
        <div>
          <h2 className="font-display text-sm font-semibold text-ink">Confidentialité des données</h2>
          <p className="mt-1 text-sm leading-6 text-text-2">{CONSULTING_DISCLAIMER}</p>
        </div>
      </Reveal>

      <div className="mt-10 space-y-12">
        {CONSULTING_CLIENTS.map((client, index) => (
          <Reveal
            key={client.slug}
            className={index === 0 ? "" : "border-t border-line pt-10"}
          >
            <section aria-labelledby={`consulting-client-${client.slug}`}>
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
                <ConsultingClientLogo client={client} />
                <div>
                  <h2
                    id={`consulting-client-${client.slug}`}
                    className="font-display text-xl font-semibold text-ink"
                  >
                    {client.name}
                  </h2>
                  <p className="mt-1 inline-flex items-center gap-1.5 text-sm text-text-2">
                    <BarChart3 size={16} aria-hidden="true" className="text-brand" />
                    Rapports Power BI
                  </p>
                </div>
              </div>

              <RevealGroup className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
                {client.reports.map((report) => (
                  <RevealItem key={report.title} className="h-full">
                    <PowerBIEmbed report={report} />
                  </RevealItem>
                ))}
              </RevealGroup>
            </section>
          </Reveal>
        ))}
      </div>
    </div>
  );
}
