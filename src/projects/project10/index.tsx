import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "inclusion-numerique-financiere",
  shortTitle: "Inclusion numérique",
  title: "Inclusion numérique & financière",
  domain: "Numérique & inclusion financière",
  family: "economie-societe-environnement",
  keywords: ["internet", "téléphonie mobile", "inclusion financière"],
  angle: "La donnée mobile et financière comme nouveau levier d'inclusion.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) d'abonnements mobiles, d'usage d'Internet, de détention de compte et de haut débit fixe dans 16 pays entre 2010 et 2024.",
  insights: [
    "Les abonnements à la téléphonie mobile dépassent largement le taux d'utilisation d'Internet dans la plupart des pays, signe d'un potentiel de croissance numérique encore important.",
    "La détention d'un compte (banque ou mobile money) progresse, mais reste mesurée par des enquêtes Findex espacées dans le temps : certaines années récentes manquent de données.",
    "Le haut débit fixe reste marginal dans la quasi-totalité du panel, l'accès à Internet passant essentiellement par le mobile.",
  ],
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
