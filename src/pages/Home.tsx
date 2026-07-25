import { Link } from "react-router-dom";
import { ArrowRight, Link2, Mail, MapPin } from "lucide-react";
import { HIGHLIGHTS, PROFILE } from "../config/content";
import Reveal, { RevealGroup, RevealItem } from "../components/Reveal";
import { DATA_TOOLS, FEATURE_POINTS, FEATURED_PROJECT_SLUGS } from "./pageData";
import { ProfileImage, ProjectCard, StatTile } from "./pageShared";

export default function Home() {
  return (
    <>
      <section className="bg-white">
        <div className="mx-auto grid max-w-7xl gap-10 px-4 py-14 sm:px-6 lg:grid-cols-[170px_1fr] lg:py-20">
          <Reveal>
            <ProfileImage />
          </Reveal>
          <Reveal delay={0.1} className="max-w-4xl">
            <p className="font-mono text-xs uppercase tracking-wide text-text-2">{PROFILE.title}</p>
            <h1 className="mt-3 max-w-3xl font-display text-4xl font-bold leading-tight tracking-tight text-ink sm:text-5xl">
              {PROFILE.fullName}
            </h1>
            <p className="mt-3 text-lg font-medium text-brand-deep">{PROFILE.tagline}</p>
            <p className="mt-5 max-w-3xl text-base leading-7 text-text-2">{PROFILE.pitch}</p>
            <p className="mt-3 flex items-center gap-1.5 text-sm text-text-2">
              <MapPin size={16} aria-hidden="true" /> {PROFILE.location}
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link
                to="/portfolio"
                className="inline-flex items-center gap-2 rounded-full bg-brand px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
              >
                Voir le portfolio <ArrowRight size={16} aria-hidden="true" />
              </Link>
              <Link
                to="/contact"
                className="rounded-full bg-surface px-5 py-2.5 text-sm font-medium text-ink transition-all duration-200 hover:scale-[1.02] hover:bg-surface-2"
              >
                Me contacter
              </Link>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="bg-brand">
        <RevealGroup className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-4 py-10 sm:grid-cols-2 sm:px-6 lg:grid-cols-4">
          {HIGHLIGHTS.map((item) => (
            <StatTile key={item.value} value={item.value} label={item.label} inverse />
          ))}
        </RevealGroup>
      </section>

      <section className="border-b border-line bg-surface">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal>
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink">Points forts</h2>
          </Reveal>
          <RevealGroup className="mt-7 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURE_POINTS.map((item) => (
              <RevealItem
                key={item.title}
                className="rounded-card border border-line bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover"
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-soft text-brand-deep">
                  <item.icon size={20} aria-hidden="true" />
                </span>
                <h3 className="mt-4 font-display text-base font-semibold text-ink">{item.title}</h3>
                <p className="mt-2 text-sm leading-6 text-text-2">{item.description}</p>
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      <section className="bg-white">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="font-display text-2xl font-bold tracking-tight text-ink">Aperçu portfolio</h2>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-text-2">
                Trois réalisations représentatives avant d'accéder aux 16 projets interactifs.
              </p>
            </div>
            <Link
              to="/portfolio"
              className="inline-flex w-fit items-center gap-2 rounded-full bg-brand px-4 py-2 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
            >
              Voir tous les projets <ArrowRight size={16} aria-hidden="true" />
            </Link>
          </Reveal>
          <RevealGroup className="mt-7 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURED_PROJECT_SLUGS.map((slug, index) => (
              <ProjectCard key={slug} slug={slug} index={index} />
            ))}
          </RevealGroup>
        </div>
      </section>

      <section className="border-y border-line bg-surface">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
          <Reveal>
            <h2 className="font-display text-xl font-bold tracking-tight text-ink">Données & outils maîtrisés</h2>
          </Reveal>
          <RevealGroup className="mt-5 flex flex-wrap gap-2">
            {DATA_TOOLS.map((tool) => (
              <RevealItem
                key={tool}
                className="rounded-full border border-line bg-white px-3 py-1.5 text-sm text-text-2"
              >
                {tool}
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      <section className="bg-white">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal className="rounded-card border border-line bg-surface p-6 sm:flex sm:items-center sm:justify-between sm:gap-8">
            <div>
              <h2 className="font-display text-xl font-bold text-ink">Un besoin de dashboard ou de data hub ?</h2>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-text-2">
                Parlons de vos indicateurs, de vos sources et de l'expérience utilisateur attendue.
              </p>
              <div className="mt-4 flex flex-wrap gap-4 text-sm text-text-2">
                <a href={PROFILE.linkedin} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 hover:text-brand">
                  <Link2 size={16} aria-hidden="true" /> LinkedIn
                </a>
                <a href={`mailto:${PROFILE.email}`} className="inline-flex items-center gap-2 hover:text-brand">
                  <Mail size={16} aria-hidden="true" /> Email
                </a>
              </div>
            </div>
            <Link
              to="/contact"
              className="mt-6 inline-flex items-center gap-2 rounded-full bg-brand px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep sm:mt-0"
            >
              Me contacter <ArrowRight size={16} aria-hidden="true" />
            </Link>
          </Reveal>
        </div>
      </section>
    </>
  );
}
