import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const OUTPUT_DIR = join(ROOT, "src", "assets", "projects");
const CREDITS_PATH = join(OUTPUT_DIR, "CREDITS.md");
const PEXELS_API = "https://api.pexels.com/v1/search";

const PROJECTS = [
  {
    slug: "sante-maternelle-neonatale",
    query: "African mother newborn health clinic",
    fallback: "mother baby Africa",
  },
  {
    slug: "nutrition-survie-enfant",
    query: "African market fresh vegetables food",
    fallback: "healthy food market Africa",
  },
  {
    slug: "education-filles-genre",
    query: "African girls classroom school",
    fallback: "students school Africa",
  },
  {
    slug: "sante-reproductive-fecondite",
    query: "African women community health center",
    fallback: "women health Africa",
  },
  {
    slug: "wash-eau-assainissement",
    query: "clean water borehole well Africa village",
    fallback: "water village Africa",
  },
  {
    slug: "populations-deplacees",
    query: "community gathering Africa humanitarian",
    fallback: "African community people",
  },
  {
    slug: "economie-croissance",
    query: "African city business district market",
    fallback: "Africa city economy",
  },
  {
    slug: "agriculture-securite-alimentaire",
    query: "African farmer cereal field harvest",
    fallback: "farming field Africa",
  },
  {
    slug: "energie-acces-electricite",
    query: "solar panels rural Africa electricity",
    fallback: "solar energy Africa",
  },
  {
    slug: "inclusion-numerique-financiere",
    query: "African person smartphone mobile payment",
    fallback: "mobile phone Africa",
  },
  {
    slug: "emploi-jeunesse",
    query: "young African professionals office team",
    fallback: "youth working Africa",
  },
  {
    slug: "climat-environnement",
    query: "Sahel savanna landscape Africa nature",
    fallback: "African landscape nature",
  },
];

function loadEnv() {
  for (const filename of [".env.local", ".env"]) {
    const path = join(ROOT, filename);
    if (!existsSync(path)) continue;

    const lines = readFileSync(path, "utf8").split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
      const [key, ...valueParts] = trimmed.split("=");
      if (!process.env[key]) {
        process.env[key] = valueParts.join("=").replace(/^["']|["']$/g, "");
      }
    }
  }
}

async function searchPhoto(apiKey, query) {
  const url = new URL(PEXELS_API);
  url.searchParams.set("query", query);
  url.searchParams.set("orientation", "landscape");
  url.searchParams.set("per_page", "8");
  url.searchParams.set("size", "large");

  const response = await fetch(url, {
    headers: {
      Authorization: apiKey,
    },
  });

  if (!response.ok) {
    throw new Error(`Pexels search failed for "${query}": ${response.status} ${response.statusText}`);
  }

  const payload = await response.json();
  return (payload.photos || []).find((photo) => photo.width >= photo.height && photo.width >= 1000);
}

async function download(url, destination) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed for ${basename(destination)}: ${response.status} ${response.statusText}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  writeFileSync(destination, bytes);
}

async function main() {
  loadEnv();
  const apiKey = process.env.PEXELS_API_KEY;
  mkdirSync(OUTPUT_DIR, { recursive: true });

  if (!apiKey) {
    writeFileSync(
      CREDITS_PATH,
      [
        "# Crédits des vignettes projets",
        "",
        "Aucune image Pexels n'a été téléchargée : `PEXELS_API_KEY` est absent.",
        "Ajoutez la clé dans `.env` ou `.env.local`, puis relancez `npm run fetch-images`.",
        "",
        "Les cartes conservent leur dégradé de fallback tant que les fichiers `src/assets/projects/{slug}.jpg` sont absents.",
        "",
      ].join("\n"),
    );
    console.log("PEXELS_API_KEY absent. Téléchargement ignoré, fallback conservé.");
    return;
  }

  const credits = [
    "# Crédits des vignettes projets",
    "",
    "Images téléchargées via l'API Pexels et bundlées localement. Licence Pexels : usage gratuit, attribution non obligatoire.",
    "",
    "| Projet | Photographe | Source | Requête |",
    "|---|---|---|---|",
  ];

  for (const project of PROJECTS) {
    const photo = (await searchPhoto(apiKey, project.query)) || (await searchPhoto(apiKey, project.fallback));

    if (!photo) {
      console.warn(`Aucune photo paysage trouvée pour ${project.slug}.`);
      credits.push(`| ${project.slug} | n.d. | n.d. | ${project.query} / ${project.fallback} |`);
      continue;
    }

    const sourceUrl = photo.src.large2x || photo.src.large || photo.src.original;
    const destination = join(OUTPUT_DIR, `${project.slug}.jpg`);
    await download(sourceUrl, destination);

    credits.push(
      `| ${project.slug} | ${photo.photographer} | ${photo.url} | ${project.query}${photo.alt ? ` (${photo.alt})` : ""} |`,
    );
    console.log(`Téléchargé ${project.slug}.jpg - ${photo.photographer}`);
  }

  credits.push("");
  writeFileSync(CREDITS_PATH, credits.join("\n"));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
