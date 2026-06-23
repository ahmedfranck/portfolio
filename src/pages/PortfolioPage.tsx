import { useState } from "react";
import { LayoutGrid } from "lucide-react";
import Reveal, { RevealGroup, RevealItem } from "../components/Reveal";
import StreamlitProjectCard from "../components/StreamlitProjectCard";
import { STREAMLIT_PROJECTS } from "../config/content";
import { PROJECTS } from "../projects";
import { FAMILIES, type PortfolioFamily } from "./pageData";
import { PageIntro, ProjectCard } from "./pageShared";

export default function PortfolioPage() {
  const [family, setFamily] = useState<PortfolioFamily>("sante-developpement-humain");
  const isStreamlitFamily = family === "applications-data-science";
  const familyProjects = isStreamlitFamily ? [] : PROJECTS.filter((project) => project.family === family);
  const visibleCount = isStreamlitFamily ? STREAMLIT_PROJECTS.length : familyProjects.length;

  return (
    <div className="bg-white">
      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-16">
        <PageIntro
          eyebrow="Portfolio"
          title="Dashboards interactifs & applications data"
          description="Les 12 tableaux de bord restent organisés par domaine, avec une nouvelle famille dédiée aux applications Streamlit et data science."
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
          {visibleCount} projets affichés
        </Reveal>

        <RevealGroup key={family} className="mt-5 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {isStreamlitFamily
            ? STREAMLIT_PROJECTS.map((project) => (
                <RevealItem key={project.slug}>
                  <StreamlitProjectCard project={project} />
                </RevealItem>
              ))
            : familyProjects.map((project, index) => (
                <ProjectCard key={project.slug} slug={project.slug} index={index} />
              ))}
        </RevealGroup>
      </section>
    </div>
  );
}
