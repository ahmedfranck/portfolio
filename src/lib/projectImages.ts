const imageModules = import.meta.glob<{ default: string }>("../assets/projects/*.jpg", {
  eager: true,
});

export const PROJECT_IMAGE_ALT: Record<string, string> = {
  "sante-maternelle-neonatale": "Photo d'une scène de santé maternelle en clinique",
  "nutrition-survie-enfant": "Photo d'un marché alimentaire avec produits frais",
  "education-filles-genre": "Photo d'élèves en salle de classe",
  "sante-reproductive-fecondite": "Photo d'un centre de santé communautaire",
  "wash-eau-assainissement": "Photo d'un point d'eau communautaire",
  "populations-deplacees": "Photo d'une communauté réunie dans un contexte humanitaire",
  "economie-croissance": "Photo d'une scène urbaine et économique africaine",
  "agriculture-securite-alimentaire": "Photo d'un champ agricole et de récoltes",
  "energie-acces-electricite": "Photo de panneaux solaires et d'accès à l'énergie",
  "inclusion-numerique-financiere": "Photo d'un usage mobile ou paiement numérique",
  "emploi-jeunesse": "Photo de jeunes professionnels au travail",
  "climat-environnement": "Photo d'un paysage naturel africain",
};

export function getProjectImage(slug: string): string | undefined {
  const match = Object.entries(imageModules).find(([path]) => path.endsWith(`/${slug}.jpg`));
  return match?.[1].default;
}

export function getProjectImageAlt(slug: string, title: string): string {
  return PROJECT_IMAGE_ALT[slug] || `Photo illustrative du projet ${title}`;
}
