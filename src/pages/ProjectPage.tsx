import { Suspense } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, ChevronRight } from "lucide-react";
import Badge from "../components/Badge";
import Reveal from "../components/Reveal";
import { PROJECTS } from "../projects";
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
  const categoryProjects = PROJECTS.filter((item) => getDashboardPortfolioCategory(item.slug) === category);
  const categoryIndex = categoryProjects.findIndex((item) => item.slug === project.slug);
  const prev = categoryProjects[(categoryIndex - 1 + categoryProjects.length) % categoryProjects.length];
  const next = categoryProjects[(categoryIndex + 1) % categoryProjects.length];
  const Body = project.Body;

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

        <Reveal className="mb-8 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="brand">{project.domain}</Badge>
            <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-text-2">
              {categoryMeta.shortLabel}
            </span>
          </div>
          <h1 className="font-display text-2xl font-bold text-ink sm:text-3xl">{project.title}</h1>
          <p className="max-w-3xl text-base text-text-2">{project.pitch}</p>
          <p className="max-w-3xl text-sm font-medium text-brand-deep">{project.angle}</p>
        </Reveal>

        <div className="space-y-6">
          <Suspense fallback={<BodyFallback />}>
            <Body />
          </Suspense>

          <Reveal className="rounded-card border border-line bg-surface p-5">
            <h2 className="font-display text-sm font-semibold text-ink">Lecture & enseignements</h2>
            <ul className="mt-2 space-y-1.5 text-sm text-ink">
              {project.insights.map((insight, i) => (
                <li key={i} className="flex gap-2">
                  <span aria-hidden="true">•</span>
                  <span>{insight}</span>
                </li>
              ))}
            </ul>
          </Reveal>
        </div>

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
              Retour aux projets en consultance
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
