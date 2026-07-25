import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { PROFILE } from "../config/content";
import { getProjectImage } from "../lib/projectImages";
import {
  getStreamlitProjectBySlug,
  getStreamlitThumbnail,
} from "../lib/streamlitProjects";
import { getProjectBySlug } from "../projects";

const SITE_URL = "https://agbadamassi.vercel.app";
const DEFAULT_IMAGE = `${SITE_URL}/favicon.jpg`;
const DEFAULT_DESCRIPTION =
  "Senior Business Analytics & Insights à Dakar : parcours, expérience et portfolio de projets data en Afrique de l'Ouest et Centrale.";

interface SeoConfig {
  title: string;
  description: string;
  image?: string;
  contentType?: "website" | "article";
  workType?: "CreativeWork" | "SoftwareApplication";
}

const STATIC_ROUTES: Record<string, SeoConfig> = {
  "/": {
    title: `${PROFILE.fullName} | ${PROFILE.title}`,
    description: DEFAULT_DESCRIPTION,
  },
  "/expertise": {
    title: `Expertise data, dashboards & monitoring | ${PROFILE.fullName}`,
    description:
      "Services de data analytics, tableaux de bord décisionnels, monitoring, gouvernance des données et intégration web.",
  },
  "/parcours": {
    title: `Parcours, compétences & certifications | ${PROFILE.fullName}`,
    description:
      "Formation, compétences techniques, certifications et spécialisation en data visualisation, BI et suivi-évaluation.",
  },
  "/experience": {
    title: `Expérience professionnelle | ${PROFILE.fullName}`,
    description:
      "Plus de 8 ans d'expérience en business analytics, performance, reporting et data visualisation en Afrique de l'Ouest.",
  },
  "/portfolio": {
    title: `Portfolio data & dashboards | ${PROFILE.fullName}`,
    description:
      "Seize projets interactifs réalisés en consultance, dans le cadre d'études et en réponse à des appels d'offres.",
  },
  "/contact": {
    title: `Contact | ${PROFILE.fullName}`,
    description:
      "Échangeons sur votre besoin en dashboard, data hub, architecture de données, suivi-évaluation ou application analytique.",
  },
};

function toAbsoluteUrl(path?: string) {
  if (!path) return DEFAULT_IMAGE;
  return new URL(path, SITE_URL).toString();
}

function getSeoConfig(pathname: string): SeoConfig {
  if (pathname.startsWith("/portfolio/app/")) {
    const slug = pathname.split("/").filter(Boolean).at(-1);
    const project = slug ? getStreamlitProjectBySlug(slug) : undefined;
    if (project) {
      return {
        title: `${project.title} | Portfolio ${PROFILE.fullName}`,
        description: project.blurb,
        image: toAbsoluteUrl(getStreamlitThumbnail(project)),
        contentType: "article",
        workType: "SoftwareApplication",
      };
    }
  }

  if (pathname.startsWith("/portfolio/")) {
    const slug = pathname.split("/").filter(Boolean).at(-1);
    const project = slug ? getProjectBySlug(slug) : undefined;
    if (project) {
      return {
        title: `${project.title} | Portfolio ${PROFILE.fullName}`,
        description: project.pitch,
        image: toAbsoluteUrl(getProjectImage(project.slug)),
        contentType: "article",
        workType: "CreativeWork",
      };
    }
  }

  return (
    STATIC_ROUTES[pathname] ?? {
      title: `Page introuvable | ${PROFILE.fullName}`,
      description: DEFAULT_DESCRIPTION,
    }
  );
}

function setMeta(selector: string, attributes: Record<string, string>) {
  let element = document.head.querySelector<HTMLMetaElement>(selector);
  if (!element) {
    element = document.createElement("meta");
    document.head.appendChild(element);
  }
  Object.entries(attributes).forEach(([name, value]) => element!.setAttribute(name, value));
}

function setCanonical(href: string) {
  let element = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!element) {
    element = document.createElement("link");
    element.rel = "canonical";
    document.head.appendChild(element);
  }
  element.href = href;
}

export default function SeoManager() {
  const location = useLocation();

  useEffect(() => {
    const config = getSeoConfig(location.pathname);
    const canonical = `${SITE_URL}${location.pathname === "/" ? "/" : location.pathname}`;
    const image = config.image ?? DEFAULT_IMAGE;

    document.title = config.title;
    setCanonical(canonical);
    setMeta('meta[name="description"]', { name: "description", content: config.description });
    setMeta('meta[property="og:title"]', { property: "og:title", content: config.title });
    setMeta('meta[property="og:description"]', { property: "og:description", content: config.description });
    setMeta('meta[property="og:type"]', { property: "og:type", content: config.contentType ?? "website" });
    setMeta('meta[property="og:url"]', { property: "og:url", content: canonical });
    setMeta('meta[property="og:image"]', { property: "og:image", content: image });
    setMeta('meta[name="twitter:card"]', { name: "twitter:card", content: "summary_large_image" });
    setMeta('meta[name="twitter:title"]', { name: "twitter:title", content: config.title });
    setMeta('meta[name="twitter:description"]', { name: "twitter:description", content: config.description });
    setMeta('meta[name="twitter:image"]', { name: "twitter:image", content: image });

    const personId = `${SITE_URL}/#person`;
    const graph: Record<string, unknown>[] = [
      {
        "@type": "Person",
        "@id": personId,
        name: PROFILE.fullName,
        jobTitle: PROFILE.title,
        url: SITE_URL,
        image: DEFAULT_IMAGE,
        email: `mailto:${PROFILE.email}`,
        sameAs: [PROFILE.linkedin],
        address: {
          "@type": "PostalAddress",
          addressLocality: "Dakar",
          addressCountry: "SN",
        },
      },
      {
        "@type": "WebPage",
        "@id": `${canonical}#webpage`,
        url: canonical,
        name: config.title,
        description: config.description,
        inLanguage: "fr",
        about: { "@id": personId },
      },
    ];

    if (config.workType) {
      graph.push({
        "@type": config.workType,
        name: config.title.split(" | ")[0],
        description: config.description,
        url: canonical,
        image,
        creator: { "@id": personId },
      });
    }

    let structuredData = document.head.querySelector<HTMLScriptElement>("#site-structured-data");
    if (!structuredData) {
      structuredData = document.createElement("script");
      structuredData.id = "site-structured-data";
      structuredData.type = "application/ld+json";
      document.head.appendChild(structuredData);
    }
    structuredData.textContent = JSON.stringify({
      "@context": "https://schema.org",
      "@graph": graph,
    });
  }, [location.pathname]);

  return null;
}
