const logoModules = import.meta.glob("../assets/logos/*.{svg,png,jpg,jpeg}", {
  eager: true,
  import: "default",
}) as Record<string, string>;

/**
 * Résout le chemin d'un logo déposé sous src/assets/logos/<filename>.
 * Retourne undefined si le fichier n'existe pas encore (fallback monogramme côté appelant).
 */
export function getEmployerLogo(filename: string): string | undefined {
  const entry = Object.entries(logoModules).find(([path]) => path.endsWith(`/${filename}`));
  return entry?.[1];
}
