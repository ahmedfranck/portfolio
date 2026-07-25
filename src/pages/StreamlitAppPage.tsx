import { Link, Navigate, useParams } from "react-router-dom";
import { ArrowLeft, Code2, ExternalLink, MonitorUp, ChevronRight } from "lucide-react";
import Badge from "../components/Badge";
import Reveal from "../components/Reveal";
import {
  getEmbedUrl,
  getStreamlitProjectBySlug,
  getStreamlitThumbnail,
  hasUsableUrl,
} from "../lib/streamlitProjects";
import { ProjectContactCta } from "./pageShared";

export default function StreamlitAppPage() {
  const { slug } = useParams<{ slug: string }>();
  const project = slug ? getStreamlitProjectBySlug(slug) : undefined;

  if (!project) {
    return <Navigate to="/portfolio" replace />;
  }

  const liveReady = hasUsableUrl(project.liveUrl);
  const repoReady = hasUsableUrl(project.repoUrl);
  const thumbnail = getStreamlitThumbnail(project);
  const embedUrl = getEmbedUrl(project.liveUrl);

  return (
    <div className="bg-surface">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <nav aria-label="Fil d'Ariane" className="mb-4 flex flex-wrap items-center gap-1.5 text-sm text-text-2">
          <Link to="/" className="transition-colors duration-200 hover:text-brand">
            Accueil
          </Link>
          <ChevronRight size={14} aria-hidden="true" />
          <Link to="/portfolio?category=etudes" className="transition-colors duration-200 hover:text-brand">
            Portfolio
          </Link>
          <ChevronRight size={14} aria-hidden="true" />
          <span className="text-ink">{project.title}</span>
        </nav>

        <Reveal className="mb-8">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="brand">Projet d'études</Badge>
            <span className="rounded-full bg-brand-soft px-3 py-1 text-xs font-medium text-brand-deep">
              Applications & data science
            </span>
            <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-text-2">
              Données {project.data}
            </span>
          </div>
          <h1 className="mt-3 font-display text-2xl font-bold text-ink sm:text-3xl">{project.title}</h1>
          <p className="mt-3 max-w-3xl text-base leading-7 text-text-2">{project.blurb}</p>

          <div className="mt-4 flex flex-wrap gap-1.5">
            {project.tags.map((tag) => (
              <span key={tag} className="rounded-full bg-white px-2.5 py-1 text-xs text-text-2">
                {tag}
              </span>
            ))}
            {project.tech.map((tech) => (
              <span key={tech} className="rounded-full border border-line bg-white px-2.5 py-1 text-xs text-text-2">
                {tech}
              </span>
            ))}
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            <Link
              to="/portfolio?category=etudes"
              className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-4 py-2 text-sm font-medium text-ink transition-colors duration-200 hover:border-brand hover:text-brand"
            >
              <ArrowLeft size={16} aria-hidden="true" /> Retour portfolio
            </Link>
            {liveReady ? (
              <a
                href={project.liveUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-full bg-brand px-4 py-2 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
              >
                Ouvrir en plein écran <ExternalLink size={16} aria-hidden="true" />
              </a>
            ) : (
              <span className="inline-flex cursor-not-allowed items-center gap-2 rounded-full bg-surface-2 px-4 py-2 text-sm font-medium text-text-2">
                Bientôt en ligne
              </span>
            )}
            {repoReady && (
              <a
                href={project.repoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-4 py-2 text-sm font-medium text-ink transition-colors duration-200 hover:border-brand hover:text-brand"
              >
                Code <Code2 size={16} aria-hidden="true" />
              </a>
            )}
          </div>
        </Reveal>

        {liveReady ? (
          <Reveal className="overflow-hidden rounded-card border border-line bg-white shadow-card">
            <iframe
              title={`Aperçu intégré de ${project.title}`}
              src={embedUrl}
              className="h-[820px] w-full"
              loading="lazy"
            />
          </Reveal>
        ) : (
          <Reveal className="rounded-card border border-line bg-white p-6 shadow-card">
            {thumbnail ? (
              <img
                src={thumbnail}
                alt={`Capture de l'application ${project.title}`}
                className="max-h-[520px] w-full rounded-card object-cover"
              />
            ) : (
              <div className="flex min-h-[320px] flex-col items-center justify-center gap-3 rounded-card bg-gradient-to-br from-brand-deep via-brand to-teal text-white">
                <MonitorUp size={42} aria-hidden="true" />
                <p className="text-sm font-medium">Aperçu intégré disponible dès que l'URL live sera renseignée.</p>
              </div>
            )}
          </Reveal>
        )}

        <ProjectContactCta
          title="Une application data ou un prototype à développer ?"
          description="Échangeons sur le cas d'usage, les données disponibles et le niveau d'interactivité attendu."
        />
      </div>
    </div>
  );
}
