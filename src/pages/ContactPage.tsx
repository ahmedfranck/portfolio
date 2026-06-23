import { Link2, Mail, MapPin, Phone } from "lucide-react";
import ContactForm from "../components/ContactForm";
import Reveal from "../components/Reveal";
import { PROFILE } from "../config/content";
import { PageIntro } from "./pageShared";

export default function ContactPage() {
  return (
    <div className="bg-surface">
      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-16">
        <PageIntro
          eyebrow="Contact"
          title="Échanger sur un projet data ou dashboard"
          description="Décrivez votre besoin : cadrage, indicateurs, architecture de données, UX/UI ou mise en production."
        />

        <div className="mt-8 grid grid-cols-1 gap-10 lg:grid-cols-2">
          <Reveal>
            <div className="flex flex-col gap-3 text-text-2">
              <a
                href={`mailto:${PROFILE.email}`}
                className="group inline-flex w-fit items-center gap-2 transition-colors duration-200 hover:text-brand"
              >
                <Mail size={16} aria-hidden="true" /> {PROFILE.email}
              </a>
              <a
                href={`tel:${PROFILE.phone.replace(/\s/g, "")}`}
                className="group inline-flex w-fit items-center gap-2 transition-colors duration-200 hover:text-brand"
              >
                <Phone size={16} aria-hidden="true" /> {PROFILE.phone}
              </a>
              <a
                href={PROFILE.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex w-fit items-center gap-2 transition-colors duration-200 hover:text-brand"
              >
                <Link2 size={16} aria-hidden="true" /> Profil LinkedIn
              </a>
              <span className="flex items-center gap-2">
                <MapPin size={16} aria-hidden="true" /> {PROFILE.location}
              </span>
            </div>

            <h2 className="mt-10 font-mono text-xs font-semibold uppercase tracking-wide text-text-2">
              Formulaire de contact
            </h2>
            <p className="mt-2 text-sm text-text-2">Décrivez votre besoin, je vous répondrai dans les meilleurs délais.</p>
          </Reveal>

          <Reveal delay={0.05}>
            <ContactForm />
          </Reveal>
        </div>
      </section>
    </div>
  );
}
