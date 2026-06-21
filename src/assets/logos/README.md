# Logos employeurs

Dépose ici les fichiers `.svg`/`.png` officiels sous ces noms exacts pour qu'ils remplacent
automatiquement le monogramme affiché par défaut dans la section Expérience :

- `myagro.svg` — myAgro Farms
- `konecta-deliveroo.svg` — Konecta (Deliveroo)
- `foundever.svg` — Foundever
- `societe-generale.svg` — Société Générale (SGABS)
- `webhelp.svg` — Webhelp

Aucune autre modification n'est nécessaire : `src/lib/employerLogos.ts` les détecte via
`import.meta.glob` au build. En leur absence, `EmployerLogo` affiche un monogramme
(`--brand-soft` / `--brand`) sans erreur.
