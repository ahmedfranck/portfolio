import { STREAMLIT_PROJECTS } from "../config/content";

export type StreamlitProject = (typeof STREAMLIT_PROJECTS)[number];

const thumbnails = import.meta.glob("../assets/projects/streamlit/*.png", {
  eager: true,
  import: "default",
  query: "?url",
}) as Record<string, string>;

export function getStreamlitProjectBySlug(slug: string) {
  return STREAMLIT_PROJECTS.find((project) => project.slug === slug);
}

export function getStreamlitThumbnail(project: StreamlitProject) {
  const fileName = project.thumb.split("/").pop();
  if (!fileName) return undefined;
  return thumbnails[`../assets/projects/streamlit/${fileName}`];
}

export function hasUsableUrl(url: string) {
  const value = url.trim();
  return value.startsWith("https://") || value.startsWith("http://");
}

export function getEmbedUrl(url: string) {
  if (!hasUsableUrl(url)) return "";
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}embed=true`;
}
