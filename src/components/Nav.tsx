import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import { Menu, User, X } from "lucide-react";
import profilePhoto from "../assets/profile.jpg";
import { PROFILE } from "../config/content";

const NAV_LINKS = [
  { path: "/", label: "Découverte" },
  { path: "/expertise", label: "Expertise" },
  { path: "/parcours", label: "Parcours" },
  { path: "/experience", label: "Expérience" },
  { path: "/portfolio", label: "Portfolio" },
  { path: "/contact", label: "Contact" },
];

function isActivePath(currentPath: string, path: string) {
  if (path === "/portfolio") {
    return currentPath === "/portfolio" || currentPath.startsWith("/portfolio/");
  }
  return currentPath === path;
}

export default function Nav() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const reduceMotion = useReducedMotion();

  return (
    <header className="sticky top-0 z-50 border-b border-line bg-white/95 backdrop-blur">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        <Link
          to="/"
          className="flex min-w-0 items-center gap-3 font-display font-medium text-ink transition-colors duration-200 hover:text-brand"
          onClick={() => setOpen(false)}
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
            const isActive = isActivePath(location.pathname, link.path);
            return (
              <li key={link.path} className="relative">
                <Link
                  to={link.path}
                  aria-current={isActive ? "page" : undefined}
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
              to="/contact"
              className="cursor-pointer rounded-full bg-brand px-4 py-2 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
            >
              Me contacter
            </Link>
          </li>
        </ul>
      </nav>

      {open && (
        <ul className="flex flex-col gap-1 border-t border-line bg-white px-4 py-3 lg:hidden">
          {NAV_LINKS.map((link) => {
            const isActive = isActivePath(location.pathname, link.path);
            return (
              <li key={link.path}>
                <Link
                  to={link.path}
                  aria-current={isActive ? "page" : undefined}
                  className={`block cursor-pointer rounded-lg px-2 py-2 text-sm font-medium transition-colors duration-200 ${
                    isActive ? "bg-brand-soft text-brand" : "text-text-2 hover:bg-surface hover:text-brand"
                  }`}
                  onClick={() => setOpen(false)}
                >
                  {link.label}
                </Link>
              </li>
            );
          })}
          <li>
            <Link
              to="/contact"
              className="mt-2 block cursor-pointer rounded-full bg-brand px-4 py-2 text-center text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep"
              onClick={() => setOpen(false)}
            >
              Me contacter
            </Link>
          </li>
        </ul>
      )}
    </header>
  );
}
