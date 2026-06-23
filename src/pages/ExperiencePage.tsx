import { Briefcase } from "lucide-react";
import EmployerLogo from "../components/EmployerLogo";
import Reveal, { RevealGroup, RevealItem } from "../components/Reveal";
import { EXPERIENCE } from "../config/content";
import { PageIntro } from "./pageShared";

export default function ExperiencePage() {
  return (
    <div className="bg-surface">
      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-16">
        <PageIntro
          eyebrow="Expérience"
          title="Chronologie professionnelle"
          description="Expériences data, performance et reporting en environnements multi-pays, avec une forte orientation décisionnelle."
        />

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
                      <h2 className="font-display text-base font-semibold text-ink">{exp.role}</h2>
                      <p className="text-sm text-text-2">
                        {exp.org}
                        {exp.location ? ` · ${exp.location}` : ""}
                      </p>
                    </div>
                    <span className="rounded-full bg-surface px-3 py-1 font-mono text-xs font-medium text-text-2">
                      {exp.period}
                    </span>
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

        <Reveal className="mt-8 rounded-card border border-line bg-white p-5 shadow-card">
          <h2 className="flex items-center gap-2 font-display text-lg font-semibold text-ink">
            <Briefcase size={20} aria-hidden="true" className="text-brand" /> Organisations traversées
          </h2>
          <div className="mt-5 flex flex-wrap gap-3">
            {Array.from(new Map(EXPERIENCE.map((exp) => [exp.org, exp])).values()).map((exp) => (
              <div key={exp.org} className="flex items-center gap-3 rounded-full border border-line bg-surface px-3 py-2">
                <EmployerLogo logo={exp.logo} name={exp.org} />
                <span className="text-sm font-medium text-ink">{exp.org}</span>
              </div>
            ))}
          </div>
        </Reveal>
      </section>
    </div>
  );
}
