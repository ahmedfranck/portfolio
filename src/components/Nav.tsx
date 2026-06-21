import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import { Menu, User, X } from "lucide-react";
import profilePhoto from "../assets/profile.jpg";
import { PROFILE } from "../config/content";
import { useActiveSection } from "../lib/useActiveSection";

const NAV_LINKS = [
  { hash: "decouverte", label: "Découverte" },
  { hash: "expertise", label: "Expertise" },
  { hash: "parcours", label: "Parcours" },
  { hash: "experience", label: "Expérience" },
  { hash: "portfolio", label: "Portfolio" },
  { hash: "contact", label: "Contact" },
];

const NAV_HASHES = NAV_LINKS.map((link) => link.hash);

export default function Nav() {
  const [open, setOpen] = useState(false);
  const [clickedHash, setClickedHash] = useState("");
  const clickedResetRef = useRef<number | null>(null);
  const location = useLocation();
  const isHome = location.pathname === "/";
  const observedHash = useActiveSection(NAV_HASHES, isHome);
  const activeHash = clickedHash || observedHash;
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    return () => {
      if (clickedResetRef.current) {
        window.clearTimeout(clickedResetRef.current);
      }
    };
  }, []);

  function handleSectionClick(event: React.MouseEvent<HTMLAnchorElement>, hash: string) {
    setOpen(false);
    setClickedHash(hash);
    if (clickedResetRef.current) {
      window.clearTimeout(clickedResetRef.current);
    }
    clickedResetRef.current = window.setTimeout(() => setClickedHash(""), 700);

    if (!isHome) return;

    const section = document.getElementById(hash);
    if (!section) return;

    event.preventDefault();
    section.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    window.history.replaceState(null, "", `#${hash}`);
  }

  return (
    <header className="sticky top-0 z-50 border-b border-line bg-white/95 backdrop-blur">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        <Link
          to="/#decouverte"
          className="flex min-w-0 items-center gap-3 font-display font-medium text-ink transition-colors duration-200 hover:text-brand"
          onClick={(event) => handleSectionClick(event, "decouverte")}
        >
          <span className="flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-full border border-brand/40 bg-surface">
            {profilePhoto ? (
              <img src={profilePhoto} alt="" className="h-full w-full object-cover" />
            ) : (
              <User size={17} aria-hidden="true" className="text-text-2" />
            )}
          </span>
          <span className="truncate text-sm sm:max-w-none">{PROFILE.fullName}</span>
        </Link>

        <button
          className="cursor-pointer rounded-full p-2 text-ink transition-colors duration-200 hover:bg-surface hover:text-brand lg:hidden"
          aria-label={open ? "Fermer le menu" : "Ouvrir le menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>

        <ul className="hidden items-center gap-1 lg:flex">
          {NAV_LINKS.map((link) => {
            const isActive = link.hash === activeHash;
            return (
              <li key={link.hash} className="relative">
                <Link
                  to={`/#${link.hash}`}
                  aria-current={isActive ? "true" : undefined}
                  onClick={(event) => handleSectionClick(event, link.hash)}
                  className={`group relative block cursor-pointer whitespace-nowrap rounded-full px-3 py-2 text-sm font-medium transition-colors duration-200 ${
                    isActive ? "text-brand" : "text-text-2 hover:text-brand"
                  }`}
                >
                  {isActive && (
                    <motion.span
                      layoutId="navIndicator"
                      className="absolute inset-0 rounded-full bg-brand-soft"
                      transition={reduceMotion ? { duration: 0 } : { type: "spring", stiffness: 380, damping: 32 }}
                    />
                  )}
                  <span className="relative z-10">{link.label}</span>
                  {!isActive && (
                    <span
                      aria-hidden="true"
                      className="absolute inset-x-3 bottom-1 h-px origin-left scale-x-0 bg-brand transition-transform duration-200 group-hover:scale-x-100"
                    />
                  )}
                </Link>
              </li>
            );
          })}
          <li>
            <Link
              to="/#contact"
              className="cursor-pointer rounded-full bg-brand px-4 py-2 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
              onClick={(event) => handleSectionClick(event, "contact")}
            >
              Me contacter
            </Link>
          </li>
        </ul>
      </nav>

      {open && (
        <ul className="flex flex-col gap-1 border-t border-line bg-white px-4 py-3 lg:hidden">
          {NAV_LINKS.map((link) => {
            const isActive = link.hash === activeHash;
            return (
              <li key={link.hash}>
                <Link
                  to={`/#${link.hash}`}
                  aria-current={isActive ? "true" : undefined}
                  className={`block cursor-pointer rounded-lg px-2 py-2 text-sm font-medium transition-colors duration-200 ${
                    isActive ? "bg-brand-soft text-brand" : "text-text-2 hover:bg-surface hover:text-brand"
                  }`}
                  onClick={(event) => handleSectionClick(event, link.hash)}
                >
                  {link.label}
                </Link>
              </li>
            );
          })}
          <li>
            <Link
              to="/#contact"
              className="mt-2 block cursor-pointer rounded-full bg-brand px-4 py-2 text-center text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
              onClick={(event) => handleSectionClick(event, "contact")}
            >
              Me contacter
            </Link>
          </li>
        </ul>
      )}
    </header>
  );
}
