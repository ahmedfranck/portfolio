export const PROFILE = {
  fullName: "Ahmed Franck-Aubin GBADAMASSI",
  title: "Senior Business Analytics & Insights",
  tagline: "Performance Groupe · Data-Driven Strategy · Web Developer",
  location: "Dakar, Sénégal",
  email: "ahmedgbadamassi@gmail.com",
  phone: "+221 77 332 70 31",
  linkedin: "https://www.linkedin.com/in/afag/",
  photo: "/src/assets/profile.jpg",
  pitch:
    "Consultant senior, je cumule plus de 8 ans d'expérience dans le traitement, l'analyse et la visualisation de données décisionnelles, avec une spécialisation sur les projets de santé publique, de suivi-évaluation et de développement en Afrique de l'Ouest. Ma double compétence data et web me permet de conduire un projet de bout en bout : cadrage fonctionnel, architecture des données, sélection et documentation des indicateurs, maquettes UX/UI, développement de tableaux de bord, intégration CMS, formation et support post-lancement.",
  bio:
    "Référent analytique avec 8+ ans d'expérience en Business Analytics, Business Intelligence et pilotage de la performance dans des environnements multi-pays en Afrique de l'Ouest. J'accompagne les directions métiers et le COMEX dans la transformation de la donnée en décisions, de la modélisation à la restitution. Maîtrise avancée de SQL (PostgreSQL), Power BI, Tableau, Dataiku, Excel avancé et Python. Bilingue français-anglais, basé à Dakar.",
};

export const HIGHLIGHTS = [
  { value: "8+ ans", label: "d'expérience data & performance" },
  { value: "3 pays · 350 000+", label: "clients pilotés (reporting COMEX)" },
  { value: "-50 %", label: "de temps de reporting (automatisation ETL)" },
  { value: "+15 % / +20 %", label: "de revenus / d'efficacité (analyses d'impact)" },
];

export interface ExperienceItem {
  role: string;
  org: string;
  period: string;
  location: string;
  /** Nom de fichier sous src/assets/logos/ (svg/png). Si absent ou introuvable : monogramme en fallback. */
  logo?: string;
  points: string[];
}

export const EXPERIENCE: ExperienceItem[] = [
  {
    role: "Business Performance Analyst",
    org: "myAgro Farms",
    period: "Juil. 2025 - Présent",
    location: "Thiès, Sénégal",
    logo: "myagro.svg",
    points: [
      "Conception et pilotage des KPIs stratégiques pour le reporting COMEX : performance par pays (Sénégal, Mali, Côte d'Ivoire), division, client et produit.",
      "Analyses de performance multi-pays pour 350 000+ clients : opportunités commerciales, clients dormants, potentiel de conversion, pipeline.",
      "Analyses prédictives (scoring, forecasting) pour anticiper les comportements clients et optimiser l'allocation des ressources terrain.",
      "Analyse des causes racines des écarts de performance ; recommandations factuelles aux directions ; formation à la culture data-driven.",
    ],
  },
  {
    role: "Senior Business Analyst",
    org: "myAgro Farms",
    period: "Juil. 2023 - Juil. 2025",
    location: "Thiès, Sénégal",
    logo: "myagro.svg",
    points: [
      "Pilotage de la roadmap des dashboards (Tableau, Dataiku) ; standardisation des définitions et cohérence des indicateurs.",
      "Pipelines ETL en Python (pandas) consolidant des données multi-sources : -50 % de temps de reporting, -30 % d'incohérences.",
      "Études d'impact (pricing, lancements, portefeuille) : +20 % d'efficacité opérationnelle, +15 % de revenus.",
      "Gouvernance des données (règles métier, traçabilité, qualité) ; mentoring de 3 analystes juniors.",
    ],
  },
  {
    role: "WFM Manager Analyst Consultant",
    org: "Konecta (Deliveroo)",
    period: "Fév. 2025 - Mars 2026",
    location: "Madagascar",
    logo: "konecta-deliveroo.svg",
    points: [
      "Dashboards automatisés de suivi des KPIs opérationnels (SLA, productivité, adhérence) ; analyses ad hoc pour les directions.",
      "Analyse de variance et identification des drivers de performance ; recommandations d'optimisation.",
      "Standardisation du reporting régional via solutions MIS (Excel avancé, VBA).",
    ],
  },
  {
    role: "Data Visualization & Monitoring Consultant",
    org: "Foundever",
    period: "Mai 2025 - Oct. 2025",
    location: "",
    logo: "foundever.svg",
    points: [
      "Production de tableaux de bord (Power BI, Tableau) pour le suivi de programmes de performance et la coordination régionale.",
      "Analyse de données d'enquêtes et de bases partenaires ; indicateurs de suivi des progrès, financements et résultats.",
      "Procédures de contrôle qualité et de validation des données.",
    ],
  },
  {
    role: "Regional Workforce Manager (Sénégal & Côte d'Ivoire)",
    org: "Foundever",
    period: "Jan. 2022 - Juil. 2023",
    location: "",
    logo: "foundever.svg",
    points: [
      "Pilotage de la performance multi-pays : KPIs stratégiques, reporting au management, analyses par pays et division.",
      "Modélisation prévisionnelle (forecasting) : +15 % de productivité.",
      "Collaboration cross-fonctionnelle (RH, Finance, Opérations) en environnement multiculturel.",
    ],
  },
  {
    role: "Reporting Engineer",
    org: "Société Générale (SGABS)",
    period: "Oct. 2021 - Déc. 2021",
    location: "Abidjan, Côte d'Ivoire",
    logo: "societe-generale.svg",
    points: [
      "Outils de visualisation pour le pilotage de la performance en environnement B2B bancaire : +35 % de rapidité de décision.",
      "Standardisation des définitions d'indicateurs ; animation transverse et formation des équipes.",
    ],
  },
  {
    role: "Senior Real Time Analyst",
    org: "Webhelp",
    period: "Mai 2018 - Août 2021",
    location: "Abidjan, Côte d'Ivoire",
    logo: "webhelp.svg",
    points: [
      "Analyses prédictives et reporting hebdomadaire pour trois lignes d'activité ; détection de tendances et signaux faibles.",
      "Encadrement de 3 analystes ; processus de reporting standardisés et promotion de la culture data.",
    ],
  },
];

export interface EducationItem {
  title: string;
  org: string;
  period: string;
}

export const EDUCATION: EducationItem[] = [
  { title: "MBA - Administration & Gestion d'entreprise", org: "CESAG, Sénégal", period: "2026 - Présent" },
  { title: "BSc Business Administration", org: "University of the People (Californie, USA)", period: "2022 - 2026" },
  { title: "DUT Logistique & Transport", org: "Université Adama Sanogo, Côte d'Ivoire", period: "2013 - 2015" },
];

export const CERTIFICATIONS: string[] = [
  "Google Project Management Professional Certificate (2026)",
  "Business Intelligence & Data Analyst - BIDA (CFI)",
  "Google Data Analytics Professional Certificate (2021)",
  "IBM Applied Data Science Professional Certificate (2022)",
  "IBM Data Analyst Professional Certificate (2022)",
  "IIBA - Certified Business Data Analyst (CBDA)",
  "IIBA - Entry Certificate in Business Analysis (ECBA)",
  "Lean Six Sigma Yellow Belt",
  "Scrum Fundamentals Certified",
];

export interface LanguageItem {
  name: string;
  level: string;
}

export const LANGUAGES: LanguageItem[] = [
  { name: "Français", level: "Courant (langue de travail)" },
  { name: "Anglais", level: "Professionnel" },
];

export const SKILLS: string[] = [
  "SQL & modélisation (PostgreSQL, schémas en étoile)",
  "BI & visualisation (Power BI/DAX, Tableau, Dataiku, Looker)",
  "Excel avancé (Power Query, TCD, PowerPivot, VBA)",
  "Python (pandas, NumPy, scikit-learn, ETL)",
  "Analytics avancés (forecasting, scoring, causes racines)",
  "KPIs commerciaux & métier (ventes, marge, funnel CRM)",
  "Gouvernance data (qualité, traçabilité, documentation)",
  "Data storytelling, formation & conduite du changement",
];

export const WEB_DEV = {
  title: "Développement web & intégration",
  items: [
    "HTML, CSS, JavaScript, Chart.js, D3.js, Plotly.",
    "Intégration WordPress / CMS et interfaces d'administration simples.",
    "Connexion API REST et ingestion de fichiers CSV, Excel, JSON.",
    "Déploiement de prototypes web et modules de visualisation.",
  ],
};

export const MONITORING = {
  title: "Data Visualization & Monitoring - santé publique / PF / SSR",
  groups: [
    {
      h: "Analyse de données & indicateurs",
      items: [
        "Définition & documentation de KPIs santé publique / PF / SSR",
        "Nettoyage, consolidation et normalisation de données multi-sources",
        "Matrices sources / fréquence / méthodologie / couverture ; dictionnaires de données",
      ],
    },
    {
      h: "Visualisation & data storytelling",
      items: [
        "Dashboards régionaux & nationaux, filtres dynamiques, comparateurs",
        "Cartes interactives, séries temporelles, trackers de progression",
        "Exports CSV / Excel / PDF ; fiches pays interactives",
      ],
    },
    {
      h: "Développement web & intégration",
      items: [
        "HTML, CSS, JavaScript, Chart.js, D3.js, Plotly",
        "Intégration WordPress / CMS ; API REST ; ingestion CSV/Excel/JSON",
        "Déploiement de prototypes web et modules de visualisation",
      ],
    },
    {
      h: "Gestion de projet & co-construction",
      items: [
        "Ateliers de cadrage avec équipes métier et partenaires",
        "Cahiers des charges fonctionnels, backlogs, priorisation",
        "Formation utilisateurs et accompagnement post-lancement",
      ],
    },
  ],
};

export const STREAMLIT_PROJECTS = [
  {
    slug: "fintech-cockpit",
    title: "Fintech Performance Cockpit",
    blurb:
      "Cockpit décisionnel unifié : Ventes, Opérations, Risque, Finance, Succès client et vue Conseil — une source de vérité unique.",
    tags: ["BI", "Finance", "KPI"],
    tech: ["Streamlit", "Python", "Plotly", "pandas"],
    data: "démo",
    liveUrl: "",
    repoUrl: "",
    thumb: "streamlit/fintech-cockpit.png",
  },
  {
    slug: "transport-ops",
    title: "Transport Operational Dashboard",
    blurb:
      "Pilotage opérationnel du transport : ponctualité, revenus/coûts, carte GPS, détection d'anomalies (IsolationForest) et prévision (Holt-Winters).",
    tags: ["Opérations", "Machine Learning", "Prévision"],
    tech: ["Streamlit", "scikit-learn", "statsmodels", "Plotly", "pydeck"],
    data: "démo",
    liveUrl: "",
    repoUrl: "",
    thumb: "streamlit/transport-ops.png",
  },
  {
    slug: "customer-segmentation",
    title: "Customer Segmentation Dashboard",
    blurb:
      "Segmentation client par K-Means : clusters, valeur vie client (CLV) et préférences produit, avec contrôle du nombre de segments.",
    tags: ["Marketing", "Machine Learning", "Clustering"],
    tech: ["Streamlit", "scikit-learn", "Altair", "pandas"],
    data: "démo",
    liveUrl: "",
    repoUrl: "",
    thumb: "streamlit/customer-segmentation.png",
  },
  {
    slug: "chess-intelligence",
    title: "Chess Intelligence Dashboard",
    blurb:
      "Analyse de la performance d'un joueur d'échecs : historique Elo, ouvertures et heatmap, alimentés en temps réel via API.",
    tags: ["API", "Analytics", "Sport"],
    tech: ["Streamlit", "Plotly", "requests", "API échecs"],
    data: "réelle (API)",
    liveUrl: "",
    repoUrl: "",
    thumb: "streamlit/chess-intelligence.png",
  },
] as const;
