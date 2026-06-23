import { Link } from "react-router-dom";
import { BarChart3, BrainCircuit, Code2, ExternalLink, ShipWheel, Trophy, type LucideIcon } from "lucide-react";
import { getStreamlitThumbnail, hasUsableUrl, type StreamlitProject } from "../lib/streamlitProjects";

const ICONS: Record<string, LucideIcon> = {
  "fintech-cockpit": BarChart3,
  "transport-ops": ShipWheel,
  "customer-segmentation": BrainCircuit,
  "chess-intelligence": Trophy,
};

export default function StreamlitProjectCard({ project }: { project: StreamlitProject }) {
  const thumbnail = getStreamlitThumbnail(project);
  const liveReady = hasUsableUrl(project.liveUrl);
  const repoReady = hasUsableUrl(project.repoUrl);
  const Icon = ICONS[project.slug] ?? Code2;

  return (
    <article className="group flex h-full flex-col overflow-hidden rounded-card border border-line bg-white transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover">
      <Link to={`/portfolio/app/${project.slug}`} className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-brand">
        <div className="relative aspect-video overflow-hidden bg-surface">
          {thumbnail ? (
            <img
              src={thumbnail}
              alt={`Capture de l'application ${project.title}`}
              loading="lazy"
              className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-brand-deep via-brand to-teal">
              <div className="flex flex-col items-center gap-3 text-white">
                <span className="flex h-14 w-14 items-center justify-center rounded-full bg-white/18">
                  <Icon size={28} aria-hidden="true" />
                </span>
                <span className="rounded-full bg-white/18 px-3 py-1 text-xs font-medium">Streamlit · Python</span>
              </div>
            </div>
          )}
          <div
            className="absolute inset-0 bg-gradient-to-t from-[rgba(20,22,27,.58)] via-[rgba(20,22,27,.12)] to-transparent"
            aria-hidden="true"
          />
          <div className="absolute left-4 top-4 rounded-full bg-white/95 px-3 py-1 text-xs font-medium text-ink shadow-sm">
            Applications & data science
          </div>
          <div className="absolute bottom-4 right-4 rounded-full bg-brand-soft px-3 py-1 text-xs font-medium text-brand-deep">
            Données {project.data}
          </div>
        </div>
      </Link>

      <div className="flex flex-1 flex-col p-4">
        <div className="flex flex-wrap gap-1.5">
          {project.tags.map((tag) => (
            <span key={tag} className="rounded-full bg-surface px-2.5 py-1 text-xs text-text-2">
              {tag}
            </span>
          ))}
        </div>

        <h2 className="mt-3 font-display text-base font-semibold leading-6 text-ink">{project.title}</h2>
        <p className="mt-2 flex-1 text-sm leading-6 text-text-2">{project.blurb}</p>

        <div className="mt-4 flex flex-wrap gap-1.5">
          {project.tech.map((tech) => (
            <span key={tech} className="rounded-full border border-line bg-white px-2.5 py-1 text-xs text-text-2">
              {tech}
            </span>
          ))}
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          {liveReady ? (
            <a
              href={project.liveUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-full bg-brand px-4 py-2 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
            >
              Ouvrir l'application <ExternalLink size={14} aria-hidden="true" />
            </a>
          ) : (
            <span
              aria-disabled="true"
              className="inline-flex cursor-not-allowed items-center gap-1.5 rounded-full bg-surface-2 px-4 py-2 text-sm font-medium text-text-2"
            >
              Bientôt en ligne
            </span>
          )}

          {repoReady && (
            <a
              href={project.repoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-full border border-line bg-white px-4 py-2 text-sm font-medium text-ink transition-colors duration-200 hover:border-brand hover:text-brand"
            >
              Code <Code2 size={14} aria-hidden="true" />
            </a>
          )}

          <Link
            to={`/portfolio/app/${project.slug}`}
            className="inline-flex items-center gap-1.5 rounded-full border border-line bg-white px-4 py-2 text-sm font-medium text-ink transition-colors duration-200 hover:border-brand hover:text-brand"
          >
            Détail
          </Link>
        </div>
      </div>
    </article>
  );
}
