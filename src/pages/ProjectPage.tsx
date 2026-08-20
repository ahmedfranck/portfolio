import { Suspense } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, ChevronRight } from "lucide-react";
import Badge from "../components/Badge";
import DashboardMasthead from "../components/DashboardMasthead";
import Reveal from "../components/Reveal";
import { getPortfolioOrganizationByProject } from "../config/portfolioOrganizations";
import { AoThemeProvider } from "../hooks/useAoTheme";
import { PROJECTS } from "../projects";
import { AO_THEMES, type AoThemeKey } from "../themes/appelsOffres";
import { getDashboardPortfolioCategory, getPortfolioCategory } from "./pageData";
import { ProjectContactCta } from "./pageShared";

function BodyFallback() {
  return (
    <div className="flex h-64 items-center justify-center rounded-card border border-line bg-white text-sm text-text-2 shadow-card">
      Chargement du tableau de bord…
    </div>
  );
}

export default function ProjectPage() {
  const { slug } = useParams<{ slug: string }>();
  const projectIndex = PROJECTS.findIndex((p) => p.slug === slug);

  if (projectIndex === -1) {
    return <Navigate to="/portfolio" replace />;
  }

  const project = PROJECTS[projectIndex];
  const category = getDashboardPortfolioCategory(project.slug);
  const categoryMeta = getPortfolioCategory(category);
  const organization = getPortfolioOrganizationByProject(project.slug);
  const categoryProjects = PROJECTS.filter((item) => getDashboardPortfolioCategory(item.slug) === category);
  const categoryIndex = categoryProjects.findIndex((item) => item.slug === project.slug);
  const prev = categoryProjects[(categoryIndex - 1 + categoryProjects.length) % categoryProjects.length];
  const next = categoryProjects[(categoryIndex + 1) % categoryProjects.length];
  const Body = project.Body;
  const isUcpo = organization?.id === "ucpo";
  const organizationKey: AoThemeKey = isUcpo
    ? "ucpo"
    : organization?.id === "bad" || organization?.id === "unicef" || organization?.id === "pnue"
      ? organization.id
      : "unverified";
  const theme = AO_THEMES[organizationKey];
  const footerNote = "footerNote" in theme ? theme.footerNote : undefined;

  return (
    <div className="bg-surface">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <nav aria-label="Fil d'Ariane" className="mb-4 flex flex-wrap items-center gap-1.5 text-sm text-text-2">
        <Link to="/" className="transition-colors duration-200 hover:text-brand">
          Accueil
        </Link>
        <ChevronRight size={14} aria-hidden="true" />
        <Link
          to={`/portfolio?category=${category}`}
          className="transition-colors duration-200 hover:text-brand"
        >
          Portfolio
        </Link>
        <ChevronRight size={14} aria-hidden="true" />
        <span className="text-ink">{project.shortTitle}</span>
      </nav>

        <Reveal className="mb-6 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="brand">{project.domain}</Badge>
            <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-text-2">
              {categoryMeta.shortLabel}
            </span>
            {organization && (
              <span
                className="rounded-full px-3 py-1 text-xs font-medium"
                style={{ color: theme.colors.primaryDark, backgroundColor: theme.colors.soft }}
              >
                {theme.organizationDisplay}
              </span>
            )}
          </div>
          <p className="max-w-3xl text-sm font-medium text-brand-deep">{project.angle}</p>
        </Reveal>

        <AoThemeProvider themeKey={organizationKey} className="space-y-6">
          <section
            className={`dashboard-experience ${isUcpo ? "dashboard-experience--ucpo" : "dashboard-experience--standard"}`}
            data-organization={organization?.id ?? "unverified"}
            aria-label={`Tableau de bord ${project.title}`}
          >
            <DashboardMasthead project={project} organization={organization} isUcpo={isUcpo} />
            <div className="dashboard-project-content">
              <Suspense fallback={<BodyFallback />}>
                <Body />
              </Suspense>
            </div>
          </section>

          <Reveal className="dashboard-insights">
            <div>
              <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-text-2">Synthèse exécutive</p>
              <h2 className="mt-1 font-display text-lg font-semibold text-ink">Lecture & enseignements</h2>
            </div>
            <ol className="mt-4 grid gap-3 lg:grid-cols-3">
              {project.insights.map((insight, i) => (
                <li key={i} className="dashboard-insights__item">
                  <span aria-hidden="true">0{i + 1}</span>
                  <span>{insight}</span>
                </li>
              ))}
            </ol>
          </Reveal>
          {footerNote && (
            <p
              className="rounded-md border px-4 py-3 text-[10px]"
              style={{ borderColor: theme.colors.border, color: theme.colors.muted, background: theme.colors.canvas }}
            >
              {footerNote}
            </p>
          )}
        </AoThemeProvider>

        <ProjectContactCta />

        {categoryProjects.length > 1 ? (
          <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:justify-between">
            <Link
              to={`/portfolio/${prev.slug}`}
              className="flex items-center gap-2 rounded-full border border-line bg-white px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:border-brand hover:text-brand"
            >
              <ArrowLeft size={16} aria-hidden="true" />
              <span>Projet précédent : {prev.shortTitle}</span>
            </Link>
            <Link
              to={`/portfolio/${next.slug}`}
              className="flex items-center justify-end gap-2 rounded-full border border-line bg-white px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:border-brand hover:text-brand"
            >
              <span>Projet suivant : {next.shortTitle}</span>
              <ArrowRight size={16} aria-hidden="true" />
            </Link>
          </div>
        ) : (
          <div className="mt-8">
            <Link
              to={`/portfolio?category=${category}`}
              className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:border-brand hover:text-brand"
            >
              <ArrowLeft size={16} aria-hidden="true" />
              Retour aux projets d'études
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
