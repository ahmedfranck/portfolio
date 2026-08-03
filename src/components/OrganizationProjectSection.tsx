import { Building2, Droplets, Landmark, Leaf, type LucideIcon } from "lucide-react";
import type { PortfolioOrganization } from "../config/portfolioOrganizations";
import Reveal, { RevealGroup } from "./Reveal";
import { ProjectCard } from "../pages/pageShared";

const ORGANIZATION_ICONS: Record<PortfolioOrganization["id"], LucideIcon> = {
  ucpo: Building2,
  bad: Landmark,
  unicef: Droplets,
  pnue: Leaf,
};

function OrganizationIdentity({ organization }: { organization: PortfolioOrganization }) {
  if (organization.logo) {
    return (
      <div className="relative h-20 w-full max-w-[520px] overflow-hidden sm:h-28" aria-label={organization.name}>
        <img
          src={organization.logo}
          alt={`Logo ${organization.name}`}
          className="absolute left-0 top-0 h-auto w-full -translate-y-[33%]"
        />
      </div>
    );
  }

  const Icon = ORGANIZATION_ICONS[organization.id];
  return (
    <div className="flex min-w-0 items-center gap-3">
      <span
        className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-white shadow-sm"
        style={{ color: organization.accent }}
      >
        <Icon size={24} aria-hidden="true" />
      </span>
      <div className="min-w-0">
        <p className="font-display text-xl font-bold text-ink">{organization.acronym}</p>
        <p className="text-sm font-medium text-text-2">{organization.name}</p>
      </div>
    </div>
  );
}

export default function OrganizationProjectSection({ organization }: { organization: PortfolioOrganization }) {
  const projectCount = organization.projectSlugs.length;

  return (
    <section aria-labelledby={`organization-${organization.id}`} className="border-t border-line pt-8 first:border-t-0 first:pt-0">
      <Reveal>
        <div
          className="border-l-4 px-4 py-5 sm:flex sm:items-center sm:justify-between sm:gap-8 sm:px-6"
          style={{ borderColor: organization.accent, backgroundColor: organization.soft }}
        >
          <div className="min-w-0 flex-1">
            <p className="mb-2 font-mono text-[11px] font-semibold uppercase tracking-wide text-text-2">
              Organisme de référence
            </p>
            <OrganizationIdentity organization={organization} />
          </div>
          <div className="mt-4 sm:mt-0 sm:max-w-md sm:text-right">
            <h2 id={`organization-${organization.id}`} className="sr-only">
              Projets associés à {organization.name}
            </h2>
            <p className="text-sm leading-6 text-text-2">{organization.focus}</p>
            <p className="mt-2 text-sm font-semibold" style={{ color: organization.accent }}>
              {projectCount} {projectCount > 1 ? "projets" : "projet"}
            </p>
          </div>
        </div>
      </Reveal>

      <RevealGroup className="mt-5 grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-4">
        {organization.projectSlugs.map((slug, index) => (
          <ProjectCard key={slug} slug={slug} index={index} />
        ))}
      </RevealGroup>
    </section>
  );
}
