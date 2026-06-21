import { config as project1 } from "./project1";
import { config as project2 } from "./project2";
import { config as project3 } from "./project3";
import { config as project4 } from "./project4";
import { config as project5 } from "./project5";
import { config as project6 } from "./project6";
import { config as project7 } from "./project7";
import { config as project8 } from "./project8";
import { config as project9 } from "./project9";
import { config as project10 } from "./project10";
import { config as project11 } from "./project11";
import { config as project12 } from "./project12";
import type { ProjectConfig } from "./types";

export const PROJECTS: ProjectConfig[] = [
  project1,
  project2,
  project3,
  project4,
  project5,
  project6,
  project7,
  project8,
  project9,
  project10,
  project11,
  project12,
];

export function getProjectBySlug(slug: string): ProjectConfig | undefined {
  return PROJECTS.find((p) => p.slug === slug);
}
