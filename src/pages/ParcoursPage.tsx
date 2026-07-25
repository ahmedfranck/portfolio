import { Award, GraduationCap, Languages as LanguagesIcon, Route } from "lucide-react";
import Reveal from "../components/Reveal";
import { CERTIFICATIONS, EDUCATION, LANGUAGES, MONITORING, PROFILE, SKILLS } from "../config/content";
import { PageIntro } from "./pageShared";

export default function ParcoursPage() {
  return (
    <div className="bg-white">
      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-16">
        <PageIntro eyebrow="Parcours" title="Profil, formation et compétences" description={PROFILE.bio} />

        <Reveal className="mt-10 rounded-card border border-line bg-surface p-5 sm:p-6" delay={0.05}>
          <h2 className="flex items-center gap-2 font-display text-lg font-semibold text-ink">
            <Route size={20} aria-hidden="true" className="text-brand" /> {MONITORING.title}
          </h2>
          <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
            {MONITORING.groups.map((group) => (
              <div key={group.h} className="rounded-card border border-line bg-white p-4">
                <h3 className="font-medium text-ink">{group.h}</h3>
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
          <Reveal className="rounded-card border border-line bg-white p-5 shadow-card lg:col-span-2">
            <h2 className="font-mono text-xs font-semibold uppercase tracking-wide text-text-2">Compétences</h2>
            <ul className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {SKILLS.map((skill) => (
                <li key={skill} className="rounded-lg bg-surface px-3 py-2 text-sm text-ink">
                  {skill}
                </li>
              ))}
            </ul>
          </Reveal>

          <div className="space-y-6">
            <Reveal className="rounded-card border border-line bg-white p-5 shadow-card">
              <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">
                <GraduationCap size={16} aria-hidden="true" /> Formation
              </h2>
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

            <Reveal className="rounded-card border border-line bg-white p-5 shadow-card" delay={0.05}>
              <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">
                <LanguagesIcon size={16} aria-hidden="true" /> Langues
              </h2>
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

        <Reveal className="mt-6 rounded-card border border-line bg-white p-5 shadow-card">
          <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">
            <Award size={16} aria-hidden="true" /> Certifications
          </h2>
          <ul className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {CERTIFICATIONS.map((certification) => (
              <li key={certification} className="rounded-lg bg-surface px-3 py-2 text-sm text-ink">
                {certification}
              </li>
            ))}
          </ul>
        </Reveal>
      </section>
    </div>
  );
}
