import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "energie-acces-electricite",
  shortTitle: "Énergie & électricité",
  title: "Énergie & accès à l'électricité",
  domain: "Énergie",
  family: "economie-societe-environnement",
  keywords: ["accès à l'électricité", "énergies renouvelables", "cuisson propre"],
  angle: "L'accès à l'énergie, condition du développement.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) d'accès à l'électricité, de cuisson propre et de part des énergies renouvelables dans 16 pays entre 2010 et 2024.",
  insights: [
    "L'accès à l'électricité a fortement progressé sur la période dans plusieurs pays, mais des écarts considérables subsistent au sein de la région.",
    "L'accès à des combustibles de cuisson propres reste en retrait par rapport à l'accès à l'électricité dans la plupart des pays du panel.",
    "Une part élevée d'énergies renouvelables ne va pas toujours de pair avec un accès généralisé à l'électricité : elle reflète parfois un mix énergétique encore peu industrialisé.",
  ],
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
