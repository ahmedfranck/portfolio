# Ahmed Franck-Aubin GBADAMASSI — Site personnel & Portfolio Data

Site personnel présentant le parcours, l'expérience et un portfolio de **6 tableaux de bord
interactifs** de data analytics en santé publique, santé de la femme et de l'enfant, et
développement en Afrique de l'Ouest et Centrale (16 pays, 2010–2024).

La page d'accueil se parcourt en sections ancrées : **Découverte → Parcours → Expérience →
Portfolio → Contact**. Chaque projet du portfolio reste accessible sur sa propre route
(`/projets/:slug`) avec KPI, filtres, graphiques interactifs, carte, export CSV et un encart
Sources détaillé.

> **✅ Données réelles.** Tous les indicateurs proviennent de l'API ouverte de la **Banque
> mondiale (World Bank Open Data, licence CC BY-4.0)**. Aucune valeur manquante n'est fabriquée :
> les cases sans donnée réelle affichent « n.d. ». Détail complet des indicateurs, codes et années
> disponibles dans [`src/data/SOURCES.md`](src/data/SOURCES.md).

## Stack technique

- React 18 + Vite + TypeScript
- TailwindCSS
- react-router-dom (navigation + ancres de section)
- Recharts (graphiques) + react-simple-maps (cartes)
- lucide-react (icônes)
- Formulaire de contact : Netlify Forms (+ fallback `mailto:`)
- 100 % statique / client-side — aucun backend, aucune clé secrète

## Installation et lancement

```bash
npm install
npm run dev
```

L'application est servie sur `http://localhost:5173`.

## Build de production

```bash
npm run build
npm run preview   # pour prévisualiser le build localement
```

Le build est généré dans `dist/`.

## Récupérer les données réelles

```bash
npm run fetch-data
```

Ce script ([`scripts/fetch_data.mjs`](scripts/fetch_data.mjs)) interroge l'API de la Banque
mondiale pour les 16 pays et 2010–2024, puis écrit 6 fichiers JSON dans `src/data/` (committés
pour que le site fonctionne hors-ligne en build statique). Il affiche dans la console la
couverture obtenue (nombre de pays et plage d'années) pour chaque indicateur — voir
[`src/data/SOURCES.md`](src/data/SOURCES.md) pour le détail des codes utilisés.

## Personnalisation

Les informations personnelles (profil, expérience, formation, certifications, langues,
compétences) sont centralisées dans [`src/config/content.ts`](src/config/content.ts).

## Formulaire de contact

Le formulaire ([`src/components/ContactForm.tsx`](src/components/ContactForm.tsx)) est câblé pour
**Netlify Forms** : un formulaire statique miroir est présent dans [`index.html`](index.html) (requis
pour que Netlify détecte le formulaire au build, les sites React ne l'exposant pas dans le HTML
initial). Pour déployer sur un autre hébergeur, remplacer l'appel `fetch("/", ...)` dans
`ContactForm.tsx` par une soumission vers Formspree (`https://formspree.io/f/[FORMSPREE_ID]`). Un
lien `mailto:` de secours est toujours affiché sous le formulaire.

## Structure du projet

```
src/
├── components/     # Layout, Nav, Footer, KpiCard, ChartCard, SourcesPanel, ContactForm, filtres, graphiques
├── data/           # jeux de données réelles (JSON), géométries, SOURCES.md
├── pages/          # Home.tsx (sections ancrées), ProjectPage.tsx (page projet générique)
├── projects/       # config + corps de chaque projet (project1 à project6)
├── lib/            # utilitaires (formatage, indicateurs dérivés, données réelles, export CSV)
├── config/         # content.ts (profil, expérience, formation, compétences)
scripts/
└── fetch_data.mjs  # récupération des données réelles (API Banque mondiale)
```

Chaque projet (`src/projects/projectN/`) expose un `config` (métadonnées : titre, pitch,
enseignements) et un composant `Body` chargé à la demande (`React.lazy`) qui importe ses propres
données et affiche son encart Sources — le bundle initial reste léger.

## Déploiement

### Netlify

Le fichier [`netlify.toml`](netlify.toml) est déjà configuré (commande de build, dossier de
publication, redirection SPA, formulaire). Sur [netlify.com](https://app.netlify.com) :

1. « Add new site » → « Import an existing project » → connecter le dépôt.
2. Build command : `npm run build` — Publish directory : `dist` (déjà pré-rempli via `netlify.toml`).
3. Déployer. Le formulaire de contact est automatiquement détecté et fonctionnel.

### Vercel

1. Importer le dépôt sur [vercel.com](https://vercel.com).
2. Framework preset : Vite (détecté automatiquement).
3. Build command : `npm run build` — Output directory : `dist`.
4. Remplacer la logique du formulaire de contact par Formspree (voir ci-dessus), Netlify Forms
   n'étant disponible que sur Netlify.

Aucune variable d'environnement n'est nécessaire (site 100 % statique).

## Avertissement sur les données

Les indicateurs présentés proviennent de la Banque mondiale (Open Data) et sont réels. Certaines
séries ont un décalage de publication : la dernière valeur disponible par pays peut dater de
quelques années avant l'année en cours — l'année exacte est toujours indiquée dans l'encart
Sources de chaque page projet. Aucune valeur n'est jamais estimée ou interpolée pour combler une
absence de donnée.
