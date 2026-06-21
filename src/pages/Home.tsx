import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  Award,
  BadgeCheck,
  Briefcase,
  Code2,
  Database,
  GraduationCap,
  Handshake,
  Languages as LanguagesIcon,
  LayoutDashboard,
  LayoutGrid,
  Link2,
  Mail,
  MapPin,
  Phone,
  Presentation,
  Route,
  ShieldCheck,
  User,
  Workflow,
} from "lucide-react";
import Layout from "../components/Layout";
import ContactForm from "../components/ContactForm";
import CountUpText from "../components/CountUpText";
import EmployerLogo from "../components/EmployerLogo";
import Reveal, { RevealGroup, RevealItem } from "../components/Reveal";
import { PROJECTS } from "../projects";
import type { ProjectFamily } from "../projects/types";
import {
  CERTIFICATIONS,
  EDUCATION,
  EXPERIENCE,
  HIGHLIGHTS,
  LANGUAGES,
  MONITORING,
  PROFILE,
  SKILLS,
  WEB_DEV,
} from "../config/content";
import { useScrollToHash } from "../lib/useScrollToHash";
import { getProjectImage, getProjectImageAlt } from "../lib/projectImages";
import profilePhoto from "../assets/profile.jpg";

function ProfileImage() {
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

const EXPERTISE = [
  {
    icon: Database,
    title: "Observatoires & data hubs",
    description:
      "Conception de hubs de données et d'observatoires pour le suivi d'indicateurs stratégiques, de la collecte à la restitution.",
  },
  {
    icon: LayoutDashboard,
    title: "Tableaux de bord décisionnels",
    description:
      "Dashboards Power BI, Tableau et Dataiku pour le pilotage de la performance et l'aide à la décision du COMEX.",
  },
  {
    icon: Workflow,
    title: "Pipelines & gouvernance des données",
    description:
      "Pipelines ETL (Python/pandas), règles de qualité et traçabilité des sources pour des données fiables et auditables.",
  },
  {
    icon: Activity,
    title: "Monitoring santé / PF / SSR",
    description: MONITORING.groups[0].items[0],
  },
  {
    icon: Code2,
    title: "Développement web & intégration",
    description: WEB_DEV.items[0],
  },
];

const FAMILIES: { id: ProjectFamily; label: string }[] = [
  { id: "sante-developpement-humain", label: "Santé & développement humain" },
  { id: "economie-societe-environnement", label: "Économie, société & environnement" },
];

const APPROACH = [
  {
    icon: Handshake,
    title: "Co-construction",
    description: "Ateliers de cadrage avec les équipes métier et les partenaires pour définir des indicateurs partagés.",
  },
  {
    icon: ShieldCheck,
    title: "Traçabilité des sources",
    description: "Documentation des sources, fréquences et méthodologies : rien n'est affiché sans origine claire.",
  },
  {
    icon: BadgeCheck,
    title: "Qualité des données",
    description: "Procédures de contrôle qualité et de validation avant toute restitution.",
  },
  {
    icon: Presentation,
    title: "Formation & accompagnement",
    description: "Formation des utilisateurs et accompagnement post-lancement pour ancrer la culture data.",
  },
];

export default function Home() {
  useScrollToHash();
  const [family, setFamily] = useState<ProjectFamily>("sante-developpement-humain");
  const familyProjects = PROJECTS.filter((project) => project.family === family);

  return (
    <Layout>
      <section id="decouverte" className="scroll-mt-16 bg-white">
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
            <p className="mt-5 max-w-2xl text-base leading-7 text-text-2">{PROFILE.pitch}</p>
            <p className="mt-3 flex items-center gap-1.5 text-sm text-text-2">
              <MapPin size={16} aria-hidden="true" /> {PROFILE.location}
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <a
                href="#portfolio"
                className="rounded-full bg-brand px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
              >
                Voir le portfolio
              </a>
              <a
                href="#contact"
                className="rounded-full bg-surface px-5 py-2.5 text-sm font-medium text-ink transition-all duration-200 hover:scale-[1.02] hover:bg-surface-2"
              >
                Me contacter
              </a>
            </div>
          </Reveal>
        </div>

        <div className="mx-auto max-w-7xl px-4 pb-14 sm:px-6">
          <RevealGroup className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {HIGHLIGHTS.map((item) => (
              <RevealItem key={item.value} className="rounded-card border border-line bg-surface p-4">
                <CountUpText text={item.value} className="font-display text-2xl font-bold text-ink" />
                <p className="mt-1 text-sm leading-5 text-text-2">{item.label}</p>
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      <section id="expertise" className="scroll-mt-16 border-t border-line bg-surface">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal>
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink">Expertise & services</h2>
            <p className="mt-2 max-w-2xl text-sm text-text-2">
              De la collecte de la donnée à la restitution décisionnelle, en environnement multi-pays.
            </p>
          </Reveal>
          <RevealGroup className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {EXPERTISE.map((item) => (
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

          <Reveal className="mt-10" delay={0.05}>
            <h3 className="font-display text-lg font-semibold text-ink">Approche</h3>
          </Reveal>
          <RevealGroup className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {APPROACH.map((item) => (
              <RevealItem
                key={item.title}
                className="rounded-card border border-line bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover"
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-surface-2 text-ink">
                  <item.icon size={20} aria-hidden="true" />
                </span>
                <h4 className="mt-4 font-display text-sm font-semibold text-ink">{item.title}</h4>
                <p className="mt-2 text-sm leading-6 text-text-2">{item.description}</p>
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      <section className="scroll-mt-16 bg-brand">
        <RevealGroup className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-12 sm:grid-cols-2 sm:px-6 lg:grid-cols-4">
          {HIGHLIGHTS.map((item) => (
            <RevealItem key={`impact-${item.value}`} className="text-center text-white sm:text-left">
              <CountUpText text={item.value} className="font-display text-3xl font-bold" />
              <p className="mt-1 text-sm leading-5 text-white/80">{item.label}</p>
            </RevealItem>
          ))}
        </RevealGroup>
      </section>

      <section id="parcours" className="scroll-mt-16 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal>
            <h2 className="flex items-center gap-2 font-display text-2xl font-bold tracking-tight text-ink">
              <Route size={22} aria-hidden="true" className="text-brand" /> Parcours
            </h2>
            <p className="mt-4 max-w-4xl text-base leading-7 text-text-2">{PROFILE.bio}</p>
          </Reveal>

          <Reveal className="mt-10 rounded-card border border-line bg-surface p-5 sm:p-6" delay={0.05}>
            <h3 className="font-display text-lg font-semibold text-ink">{MONITORING.title}</h3>
            <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
              {MONITORING.groups.map((group) => (
                <div key={group.h} className="rounded-card border border-line bg-white p-4">
                  <h4 className="font-medium text-ink">{group.h}</h4>
                  <ul className="mt-3 space-y-2 text-sm leading-6 text-text-2">
                    {group.items.map((item) => (
                      <li key={item} className="flex gap-2">
                        <span aria-hidden="true" className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </Reveal>

          <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
            <Reveal className="rounded-card border border-line bg-white p-5 lg:col-span-2">
              <h3 className="font-mono text-xs font-semibold uppercase tracking-wide text-text-2">Compétences</h3>
              <ul className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {SKILLS.map((skill) => (
                  <li key={skill} className="rounded-lg bg-surface px-3 py-2 text-sm text-ink">
                    {skill}
                  </li>
                ))}
              </ul>

              <h4 className="mt-6 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">{WEB_DEV.title}</h4>
              <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {WEB_DEV.items.map((item) => (
                  <li key={item} className="rounded-lg bg-surface px-3 py-2 text-sm text-ink">
                    {item}
                  </li>
                ))}
              </ul>
            </Reveal>

            <div className="space-y-6">
              <Reveal className="rounded-card border border-line bg-white p-5">
                <h3 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">
                  <GraduationCap size={16} aria-hidden="true" /> Formation
                </h3>
                <ol className="mt-4 space-y-4">
                  {EDUCATION.map((ed) => (
                    <li key={ed.title}>
                      <p className="text-sm font-medium text-ink">{ed.title}</p>
                      <p className="text-sm text-text-2">{ed.org}</p>
                      <p className="text-xs text-text-2">{ed.period}</p>
                    </li>
                  ))}
                </ol>
              </Reveal>

              <Reveal className="rounded-card border border-line bg-white p-5" delay={0.05}>
                <h3 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">
                  <LanguagesIcon size={16} aria-hidden="true" /> Langues
                </h3>
                <div className="mt-4 flex flex-wrap gap-2">
                  {LANGUAGES.map((language) => (
                    <span key={language.name} className="rounded-full bg-surface px-3 py-1.5 text-sm text-text-2">
                      <span className="font-medium text-ink">{language.name}</span> · {language.level}
                    </span>
                  ))}
                </div>
              </Reveal>
            </div>
          </div>

          <Reveal className="mt-6 rounded-card border border-line bg-white p-5">
            <h3 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">
              <Award size={16} aria-hidden="true" /> Certifications
            </h3>
            <ul className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {CERTIFICATIONS.map((certification) => (
                <li key={certification} className="rounded-lg bg-surface px-3 py-2 text-sm text-ink">
                  {certification}
                </li>
              ))}
            </ul>
          </Reveal>
        </div>
      </section>

      <section id="experience" className="scroll-mt-16 border-t border-line bg-surface">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal>
            <h2 className="flex items-center gap-2 font-display text-2xl font-bold tracking-tight text-ink">
              <Briefcase size={22} aria-hidden="true" className="text-brand" /> Expérience
            </h2>
          </Reveal>
          <RevealGroup className="mt-8 space-y-4">
            {EXPERIENCE.map((exp) => (
              <RevealItem
                key={`${exp.role}-${exp.org}`}
                className="group rounded-card border border-line bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover"
              >
                <div className="flex flex-wrap items-start gap-4">
                  <EmployerLogo logo={exp.logo} name={exp.org} />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <h3 className="font-display text-base font-semibold text-ink">{exp.role}</h3>
                        <p className="text-sm text-text-2">
                          {exp.org}
                          {exp.location ? ` · ${exp.location}` : ""}
                        </p>
                      </div>
                      <span className="rounded-full bg-surface px-3 py-1 font-mono text-xs font-medium text-text-2">{exp.period}</span>
                    </div>
                    <ul className="mt-4 space-y-2 text-sm leading-6 text-ink">
                      {exp.points.map((point) => (
                        <li key={point} className="flex gap-2">
                          <span aria-hidden="true" className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
                          <span>{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      <section id="portfolio" className="scroll-mt-16 border-y border-line bg-white">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal>
            <h2 className="flex items-center gap-2 font-display text-2xl font-bold tracking-tight text-ink">
              <LayoutGrid size={22} aria-hidden="true" className="text-brand" /> Portfolio
            </h2>
            <p className="mt-2 max-w-2xl text-sm text-text-2">
              Douze tableaux de bord interactifs, multi-onglets, organisés en deux familles : santé &amp;
              développement humain, et économie, société &amp; environnement.
            </p>
          </Reveal>

          <Reveal delay={0.05} className="mt-6 flex flex-wrap gap-2">
            {FAMILIES.map((f) => {
              const isActive = f.id === family;
              return (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => setFamily(f.id)}
                  aria-pressed={isActive}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition-colors duration-200 ${
                    isActive ? "bg-brand text-white" : "bg-surface text-text-2 hover:bg-surface-2 hover:text-ink"
                  }`}
                >
                  {f.label}
                </button>
              );
            })}
          </Reveal>

          <RevealGroup key={family} className="mt-7 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {familyProjects.map((project, index) => (
              <RevealItem key={project.slug}>
                <Link
                  to={`/projets/${project.slug}`}
                  className="group block overflow-hidden rounded-card border border-line bg-white transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover"
                >
                  <div className="relative aspect-video overflow-hidden bg-surface">
                    {getProjectImage(project.slug) ? (
                      <img
                        src={getProjectImage(project.slug)}
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
                    <div className="absolute inset-0 bg-gradient-to-t from-[rgba(20,22,27,.60)] via-[rgba(20,22,27,.18)] to-transparent" aria-hidden="true" />
                    <div className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-3 p-4">
                      <span className="rounded-full bg-white/95 px-3 py-1 text-xs font-medium text-ink shadow-sm">{project.domain}</span>
                      <span className="line-clamp-2 text-right font-display text-sm font-semibold leading-5 text-white drop-shadow">
                        {project.shortTitle}
                      </span>
                    </div>
                  </div>
                  <div className="p-4">
                    <h3 className="font-display text-base font-medium leading-6 text-ink">{project.title}</h3>
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
            ))}
          </RevealGroup>
        </div>
      </section>

      <section id="contact" className="scroll-mt-16 bg-surface">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal>
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink">Contact</h2>
          </Reveal>
          <div className="mt-6 grid grid-cols-1 gap-10 lg:grid-cols-2">
            <Reveal>
              <div className="flex flex-col gap-3 text-text-2">
                <a href={`mailto:${PROFILE.email}`} className="flex items-center gap-2 hover:text-brand">
                  <Mail size={16} aria-hidden="true" /> {PROFILE.email}
                </a>
                <a href={`tel:${PROFILE.phone.replace(/\s/g, "")}`} className="flex items-center gap-2 hover:text-brand">
                  <Phone size={16} aria-hidden="true" /> {PROFILE.phone}
                </a>
                <a
                  href={PROFILE.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 hover:text-brand"
                >
                  <Link2 size={16} aria-hidden="true" /> Profil LinkedIn
                </a>
                <span className="flex items-center gap-2">
                  <MapPin size={16} aria-hidden="true" /> {PROFILE.location}
                </span>
              </div>

              <h3 className="mt-10 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">Formulaire de contact</h3>
              <p className="mt-2 text-sm text-text-2">Décrivez votre besoin, je vous répondrai dans les meilleurs délais.</p>
            </Reveal>

            <Reveal delay={0.05}>
              <ContactForm />
            </Reveal>
          </div>
        </div>
      </section>
    </Layout>
  );
}
