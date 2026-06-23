import { Link } from "react-router-dom";
import { ArrowRight, User } from "lucide-react";
import CountUpText from "../components/CountUpText";
import Reveal, { RevealItem } from "../components/Reveal";
import { PROFILE } from "../config/content";
import { getProjectImage, getProjectImageAlt } from "../lib/projectImages";
import { PROJECTS } from "../projects";
import profilePhoto from "../assets/profile.jpg";

export function ProfileImage() {
  return (
    <div
      className="flex h-[150px] w-[150px] shrink-0 items-center justify-center overflow-hidden rounded-full border-2 border-brand bg-surface"
      role="img"
      aria-label={`Photo de profil de ${PROFILE.fullName}`}
    >
      {profilePhoto ? (
        <img src={profilePhoto} alt="" className="h-full w-full object-cover" />
      ) : (
        <User size={52} aria-hidden="true" className="text-text-3" />
      )}
    </div>
  );
}

export function PageIntro({
  eyebrow,
  title,
  description,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <Reveal>
      {eyebrow && <p className="font-mono text-xs font-semibold uppercase tracking-wide text-brand">{eyebrow}</p>}
      <h1 className="mt-2 font-display text-3xl font-bold tracking-tight text-ink sm:text-4xl">{title}</h1>
      {description && <p className="mt-3 max-w-3xl text-base leading-7 text-text-2">{description}</p>}
    </Reveal>
  );
}

export function StatTile({ value, label, inverse = false }: { value: string; label: string; inverse?: boolean }) {
  return (
    <RevealItem
      className={`rounded-card p-4 transition-all duration-200 hover:-translate-y-[3px] ${
        inverse
          ? "text-white hover:bg-white/10"
          : "border border-line bg-surface text-ink shadow-sm hover:shadow-cardHover"
      }`}
    >
      <CountUpText text={value} className="font-display text-2xl font-bold sm:text-3xl" />
      <p className={`mt-1 text-sm leading-5 ${inverse ? "text-white/80" : "text-text-2"}`}>{label}</p>
    </RevealItem>
  );
}

export function ProjectCard({ slug, index = 0 }: { slug: string; index?: number }) {
  const project = PROJECTS.find((item) => item.slug === slug);

  if (!project) return null;

  const image = getProjectImage(project.slug);

  return (
    <RevealItem>
      <Link
        to={`/portfolio/${project.slug}`}
        className="group block h-full overflow-hidden rounded-card border border-line bg-white transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover"
      >
        <div className="relative aspect-video overflow-hidden bg-surface">
          {image ? (
            <img
              src={image}
              alt={getProjectImageAlt(project.slug, project.title)}
              loading="lazy"
              className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
            />
          ) : (
            <div
              className="h-full w-full transition-transform duration-300 group-hover:scale-105"
              style={{
                background:
                  index % 3 === 0
                    ? "linear-gradient(135deg, #5B4BE3 0%, #0EA5A4 100%)"
                    : index % 3 === 1
                      ? "linear-gradient(135deg, #3F32B5 0%, #F59E0B 100%)"
                      : "linear-gradient(135deg, #0EA5A4 0%, #64748B 100%)",
              }}
              aria-hidden="true"
            />
          )}
          <div
            className="absolute inset-0 bg-gradient-to-t from-[rgba(20,22,27,.60)] via-[rgba(20,22,27,.18)] to-transparent"
            aria-hidden="true"
          />
          <div className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-3 p-4">
            <span className="rounded-full bg-white/95 px-3 py-1 text-xs font-medium text-ink shadow-sm">
              {project.domain}
            </span>
            <span className="line-clamp-2 text-right font-display text-sm font-semibold leading-5 text-white drop-shadow">
              {project.shortTitle}
            </span>
          </div>
        </div>
        <div className="p-4">
          <h2 className="font-display text-base font-medium leading-6 text-ink">{project.title}</h2>
          <p className="mt-1 line-clamp-2 text-sm leading-5 text-text-2">{project.angle}</p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {project.keywords.map((keyword) => (
              <span key={keyword} className="rounded-full bg-surface px-2.5 py-1 text-xs text-text-2">
                {keyword}
              </span>
            ))}
          </div>
          <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-brand">
            Découvrir <ArrowRight size={14} className="transition-transform group-hover:translate-x-0.5" />
          </span>
        </div>
      </Link>
    </RevealItem>
  );
}
