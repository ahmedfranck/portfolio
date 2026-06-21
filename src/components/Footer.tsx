import { Link2, Mail } from "lucide-react";
import { PROFILE } from "../config/content";

export default function Footer() {
  return (
    <footer className="border-t border-line bg-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-8 text-sm text-text-2 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>
          © {new Date().getFullYear()} {PROFILE.fullName}
        </p>
        <div className="flex flex-wrap gap-4">
          <a
            href={`mailto:${PROFILE.email}`}
            className="inline-flex items-center gap-2 underline-offset-4 transition-colors duration-200 hover:text-brand hover:underline"
          >
            <Mail size={16} aria-hidden="true" /> Email
          </a>
          <a
            href={PROFILE.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 underline-offset-4 transition-colors duration-200 hover:text-brand hover:underline"
          >
            <Link2 size={16} aria-hidden="true" /> LinkedIn
          </a>
        </div>
      </div>
    </footer>
  );
}
