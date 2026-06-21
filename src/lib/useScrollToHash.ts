import { useEffect } from "react";
import { useLocation } from "react-router-dom";

/** Scrolle en douceur vers l'élément dont l'id correspond au hash de l'URL (#section). */
export function useScrollToHash() {
  const { hash } = useLocation();

  useEffect(() => {
    if (!hash) return;
    const id = hash.replace("#", "");
    const el = document.getElementById(id);
    if (el) {
      requestAnimationFrame(() => el.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  }, [hash]);
}
