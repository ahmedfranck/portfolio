import { motion, useReducedMotion } from "framer-motion";
import { LayoutGrid } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import Reveal, { RevealGroup, RevealItem } from "../components/Reveal";
import StreamlitProjectCard from "../components/StreamlitProjectCard";
import { STREAMLIT_PROJECTS } from "../config/content";
import { PROJECTS } from "../projects";
import {
  getDashboardPortfolioCategory,
  getPortfolioCategory,
  PORTFOLIO_CATEGORIES,
  type PortfolioCategory,
} from "./pageData";
import { PageIntro, ProjectCard } from "./pageShared";

const CATEGORY_IDS = new Set<PortfolioCategory>(PORTFOLIO_CATEGORIES.map((category) => category.id));

export default function PortfolioPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const reduceMotion = useReducedMotion();
  const requestedCategory = searchParams.get("category") as PortfolioCategory | null;
  const category = requestedCategory && CATEGORY_IDS.has(requestedCategory) ? requestedCategory : "consultance";
  const isStudyCategory = category === "etudes";
  const dashboardProjects = isStudyCategory
    ? []
    : PROJECTS.filter((project) => getDashboardPortfolioCategory(project.slug) === category);
  const visibleCount = isStudyCategory ? STREAMLIT_PROJECTS.length : dashboardProjects.length;
  const activeCategory = getPortfolioCategory(category);

  function selectCategory(nextCategory: PortfolioCategory) {
    setSearchParams({ category: nextCategory });
  }

  return (
    <div className="bg-white">
      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-16">
        <PageIntro
          eyebrow="Portfolio"
          title="Dashboards interactifs & applications data"
          description="Seize projets organisés selon leur contexte de réalisation : consultance, études et réponses à des appels d'offres."
        />

        <Reveal delay={0.05} className="mt-7 grid grid-cols-1 gap-2 lg:grid-cols-3">
          {PORTFOLIO_CATEGORIES.map((item) => {
            const isActive = item.id === category;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => selectCategory(item.id)}
                aria-pressed={isActive}
                className={`group relative flex min-h-12 cursor-pointer items-center gap-2 overflow-hidden rounded-card px-4 py-3 text-left text-sm font-medium transition-colors duration-200 ${
                  isActive ? "text-brand-deep" : "bg-surface text-text-2 hover:text-brand"
                }`}
              >
                {isActive && (
                  <motion.span
                    layoutId="portfolioCategoryIndicator"
                    className="absolute inset-0 border border-brand/20 bg-brand-soft"
                    transition={reduceMotion ? { duration: 0 } : { type: "spring", stiffness: 380, damping: 32 }}
                  />
                )}
                <Icon size={18} aria-hidden="true" className="relative z-10 shrink-0" />
                <span className="relative z-10">{item.label}</span>
              </button>
            );
          })}
        </Reveal>

        <Reveal className="mt-8 border-y border-line py-5" key={`${category}-summary`}>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h2 className="font-display text-xl font-semibold text-ink">{activeCategory.label}</h2>
              <p className="mt-1 max-w-3xl text-sm leading-6 text-text-2">{activeCategory.description}</p>
            </div>
            <span className="inline-flex w-fit shrink-0 items-center gap-2 rounded-full bg-surface px-3 py-1.5 text-sm font-medium text-text-2">
              <LayoutGrid size={16} aria-hidden="true" className="text-brand" />
              {visibleCount} {visibleCount > 1 ? "projets" : "projet"}
            </span>
          </div>
        </Reveal>

        <RevealGroup key={category} className="mt-6 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {isStudyCategory
            ? STREAMLIT_PROJECTS.map((project) => (
                <RevealItem key={project.slug}>
                  <StreamlitProjectCard project={project} />
                </RevealItem>
              ))
            : dashboardProjects.map((project, index) => (
                <ProjectCard key={project.slug} slug={project.slug} index={index} />
              ))}
        </RevealGroup>
      </section>
    </div>
  );
}
