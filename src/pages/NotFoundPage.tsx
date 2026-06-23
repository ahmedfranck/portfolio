import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import Reveal from "../components/Reveal";

export default function NotFoundPage() {
  return (
    <div className="bg-white">
      <section className="mx-auto max-w-3xl px-4 py-20 text-center sm:px-6">
        <Reveal>
          <p className="font-mono text-xs font-semibold uppercase tracking-wide text-brand">404</p>
          <h1 className="mt-3 font-display text-3xl font-bold text-ink">Page introuvable</h1>
          <p className="mt-3 text-sm leading-6 text-text-2">
            Cette page n'existe pas ou a été déplacée dans la nouvelle structure multi-pages.
          </p>
          <Link
            to="/"
            className="mt-7 inline-flex items-center gap-2 rounded-full bg-brand px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
          >
            <ArrowLeft size={16} aria-hidden="true" /> Retour accueil
          </Link>
        </Reveal>
      </section>
    </div>
  );
}
