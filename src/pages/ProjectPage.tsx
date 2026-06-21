import { Suspense } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, ChevronRight } from "lucide-react";
import Layout from "../components/Layout";
import Badge from "../components/Badge";
import Reveal from "../components/Reveal";
import { PROJECTS } from "../projects";

function BodyFallback() {
  return (
    <div className="flex h-64 items-center justify-center rounded-card border border-line bg-white text-sm text-text-2 shadow-card">
      Chargement du tableau de bord…
    </div>
  );
}

export default function ProjectPage() {
  const { slug } = useParams<{ slug: string }>();
  const index = PROJECTS.findIndex((p) => p.slug === slug);

  if (index === -1) {
    return <Navigate to="/" replace />;
  }

  const project = PROJECTS[index];
  const prev = PROJECTS[(index - 1 + PROJECTS.length) % PROJECTS.length];
  const next = PROJECTS[(index + 1) % PROJECTS.length];
  const Body = project.Body;

  return (
    <Layout>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <nav aria-label="Fil d'Ariane" className="mb-4 flex items-center gap-1.5 text-sm text-text-2">
          <Link to="/" className="hover:text-brand">
            Accueil
          </Link>
          <ChevronRight size={14} aria-hidden="true" />
          <span className="text-ink">{project.shortTitle}</span>
        </nav>

        <Reveal className="mb-8 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="brand">{project.domain}</Badge>
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

        <div className="mt-10 flex flex-col gap-3 border-t border-line pt-6 sm:flex-row sm:justify-between">
          <Link
            to={`/projets/${prev.slug}`}
            className="flex items-center gap-2 rounded-full border border-line bg-white px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:border-brand hover:text-brand"
          >
            <ArrowLeft size={16} aria-hidden="true" />
            <span>Projet précédent : {prev.shortTitle}</span>
          </Link>
          <Link
            to={`/projets/${next.slug}`}
            className="flex items-center justify-end gap-2 rounded-full border border-line bg-white px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:border-brand hover:text-brand"
          >
            <span>Projet suivant : {next.shortTitle}</span>
            <ArrowRight size={16} aria-hidden="true" />
          </Link>
        </div>
      </div>
    </Layout>
  );
}
