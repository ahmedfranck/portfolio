import type { ComponentType } from "react";

export type ProjectFamily = "sante-developpement-humain" | "economie-societe-environnement";

export interface ProjectConfig {
  slug: string;
  shortTitle: string;
  title: string;
  domain: string;
  family: ProjectFamily;
  keywords: string[];
  pitch: string;
  angle: string;
  insights: string[];
  hasMap: boolean;
  Body: ComponentType;
}
