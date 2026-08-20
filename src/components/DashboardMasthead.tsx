import { Activity, CalendarRange, DatabaseZap, Globe2, ShieldCheck } from "lucide-react";
import type { PortfolioOrganization } from "../config/portfolioOrganizations";
import type { ProjectConfig } from "../projects/types";

interface DashboardMastheadProps {
  project: ProjectConfig;
  organization?: PortfolioOrganization;
  isUcpo?: boolean;
}

function extractCoverage(pitch: string) {
  return pitch.match(/(\d+) pays/i)?.[1] ?? "Régional";
}

function extractPeriod(pitch: string) {
  const match = pitch.match(/(20\d{2})\s*(?:à|→|–|-)\s*(20\d{2})/i);
  return match ? `${match[1]}–${match[2]}` : "Multi-années";
}

export default function DashboardMasthead({ project, organization, isUcpo = false }: DashboardMastheadProps) {
  const coverage = extractCoverage(project.pitch);
  const period = extractPeriod(project.pitch);

  return (
    <header className="dashboard-masthead">
      <div className="dashboard-masthead__glow" aria-hidden="true" />
      <div className="relative z-10 grid gap-8 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-start">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="dashboard-masthead__eyebrow">
              <Activity size={14} aria-hidden="true" />
              {isUcpo ? "Observatoire décisionnel" : "Tableau de bord analytique"}
            </span>
            <span className="dashboard-masthead__status">
              <span className="dashboard-masthead__status-dot" aria-hidden="true" />
              Données consolidées
            </span>
          </div>
          <p className="mt-5 font-mono text-[10px] font-semibold uppercase tracking-[0.22em] text-white/55">
            {organization ? `${organization.acronym} · ${organization.name}` : "Portail analytique régional"}
          </p>
          <h1 className="dashboard-masthead__title">{project.title}</h1>
          <p className="dashboard-masthead__pitch">{project.pitch}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            {project.keywords.map((keyword) => (
              <span key={keyword} className="dashboard-masthead__keyword">
                {keyword}
              </span>
            ))}
          </div>
        </div>

        {organization?.logo && (
          <div className="dashboard-masthead__identity">
            <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-white/45">
              Organisme de référence
            </span>
            <img src={organization.logo} alt={`Logo ${organization.name}`} className="mt-3 max-h-14 max-w-[210px] object-contain" />
          </div>
        )}
      </div>

      <div className="dashboard-masthead__metrics">
        <div>
          <Globe2 size={17} aria-hidden="true" />
          <span><strong>{coverage}</strong>{coverage === "Régional" ? "Couverture" : "pays couverts"}</span>
        </div>
        <div>
          <CalendarRange size={17} aria-hidden="true" />
          <span><strong>{period}</strong>Période d'analyse</span>
        </div>
        <div>
          <DatabaseZap size={17} aria-hidden="true" />
          <span><strong>{project.keywords.length} axes</strong>Lecture croisée</span>
        </div>
        <div>
          <ShieldCheck size={17} aria-hidden="true" />
          <span><strong>Traçable</strong>Sources documentées</span>
        </div>
      </div>
    </header>
  );
}
