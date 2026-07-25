import { Code2, Database } from "lucide-react";
import Reveal, { RevealGroup, RevealItem } from "../components/Reveal";
import { MONITORING, WEB_DEV } from "../config/content";
import { APPROACH, EXPERTISE } from "./pageData";
import { PageIntro } from "./pageShared";

export default function ExpertisePage() {
  return (
    <div className="bg-surface">
      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-16">
        <PageIntro
          eyebrow="Expertise"
          title="Services data, dashboards et monitoring"
          description="De la collecte de la donnée à la restitution décisionnelle, en environnement multi-pays."
        />

        <RevealGroup className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {EXPERTISE.map((item) => (
            <RevealItem
              key={item.title}
              className="rounded-card border border-line bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover"
            >
              <span className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-soft text-brand-deep">
                <item.icon size={20} aria-hidden="true" />
              </span>
              <h2 className="mt-4 font-display text-base font-semibold text-ink">{item.title}</h2>
              <p className="mt-2 text-sm leading-6 text-text-2">{item.description}</p>
            </RevealItem>
          ))}
        </RevealGroup>
      </section>

      <section className="border-y border-line bg-white">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <Reveal>
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink">Approche</h2>
          </Reveal>
          <RevealGroup className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {APPROACH.map((item) => (
              <RevealItem
                key={item.title}
                className="rounded-card border border-line bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover"
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-surface-2 text-ink">
                  <item.icon size={20} aria-hidden="true" />
                </span>
                <h3 className="mt-4 font-display text-sm font-semibold text-ink">{item.title}</h3>
                <p className="mt-2 text-sm leading-6 text-text-2">{item.description}</p>
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-14 sm:px-6 lg:grid-cols-2">
        <Reveal className="rounded-card border border-line bg-white p-6 shadow-card">
          <span className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-soft text-brand-deep">
            <Code2 size={20} aria-hidden="true" />
          </span>
          <h2 className="mt-4 font-display text-lg font-semibold text-ink">{WEB_DEV.title}</h2>
          <ul className="mt-4 space-y-2 text-sm leading-6 text-text-2">
            {WEB_DEV.items.map((item) => (
              <li key={item} className="flex gap-2">
                <span aria-hidden="true" className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </Reveal>

        <Reveal className="rounded-card border border-line bg-white p-6 shadow-card" delay={0.05}>
          <span className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-soft text-brand-deep">
            <Database size={20} aria-hidden="true" />
          </span>
          <h2 className="mt-4 font-display text-lg font-semibold text-ink">{MONITORING.title}</h2>
          <div className="mt-4 grid grid-cols-1 gap-3">
            {MONITORING.groups
              .filter((group) => group.h !== WEB_DEV.title)
              .map((group) => (
              <div key={group.h} className="rounded-lg bg-surface p-3">
                <h3 className="text-sm font-medium text-ink">{group.h}</h3>
                <p className="mt-1 text-sm leading-6 text-text-2">{group.items[0]}</p>
              </div>
              ))}
          </div>
        </Reveal>
      </section>
    </div>
  );
}
