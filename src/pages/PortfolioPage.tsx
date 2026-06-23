import { useState } from "react";
import { LayoutGrid } from "lucide-react";
import Reveal, { RevealGroup } from "../components/Reveal";
import { PROJECTS } from "../projects";
import type { ProjectFamily } from "../projects/types";
import { FAMILIES } from "./pageData";
import { PageIntro, ProjectCard } from "./pageShared";

export default function PortfolioPage() {
  const [family, setFamily] = useState<ProjectFamily>("sante-developpement-humain");
  const familyProjects = PROJECTS.filter((project) => project.family === family);

  return (
    <div className="bg-white">
      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-16">
        <PageIntro
          eyebrow="Portfolio"
          title="12 tableaux de bord interactifs"
          description="Deux familles de projets : santé & développement humain, puis économie, société & environnement."
        />

        <Reveal delay={0.05} className="mt-6 flex flex-wrap gap-2">
          {FAMILIES.map((f) => {
            const isActive = f.id === family;
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => setFamily(f.id)}
                aria-pressed={isActive}
                className={`cursor-pointer rounded-full px-4 py-2 text-sm font-medium transition-all duration-200 ${
                  isActive ? "bg-brand text-white" : "bg-surface text-text-2 hover:scale-[1.02] hover:bg-brand-soft hover:text-brand"
                }`}
              >
                {f.label}
              </button>
            );
          })}
        </Reveal>

        <Reveal className="mt-8 flex items-center gap-2 text-sm font-medium text-text-2">
          <LayoutGrid size={16} aria-hidden="true" className="text-brand" />
          {familyProjects.length} projets affichés
        </Reveal>

        <RevealGroup key={family} className="mt-5 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {familyProjects.map((project, index) => (
            <ProjectCard key={project.slug} slug={project.slug} index={index} />
          ))}
        </RevealGroup>
      </section>
    </div>
  );
}
