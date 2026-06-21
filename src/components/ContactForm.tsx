import { useState, type FormEvent } from "react";
import { AlertCircle, CheckCircle2, Mail, Send } from "lucide-react";
import { PROFILE } from "../config/content";

type Status = "idle" | "submitting" | "success" | "error";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const FORMSUBMIT_ENDPOINT = `https://formsubmit.co/ajax/${PROFILE.email}`;

export default function ContactForm() {
  const [status, setStatus] = useState<Status>("idle");
  const [values, setValues] = useState({ name: "", email: "", message: "" });
  const [error, setError] = useState("");

  function update(field: keyof typeof values, value: string) {
    setValues((v) => ({ ...v, [field]: value }));
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (!values.name.trim() || !values.email.trim() || !values.message.trim()) {
      setError("Merci de renseigner tous les champs.");
      return;
    }
    if (!EMAIL_RE.test(values.email)) {
      setError("Adresse e-mail invalide.");
      return;
    }

    setError("");
    setStatus("submitting");
    try {
      const response = await fetch(FORMSUBMIT_ENDPOINT, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: values.name,
          email: values.email,
          message: values.message,
          _subject: `Nouveau message portfolio - ${values.name}`,
          _template: "table",
          _captcha: "false",
        }),
      });

      if (!response.ok) {
        throw new Error("Form submission failed");
      }

      setStatus("success");
      setValues({ name: "", email: "", message: "" });
    } catch {
      setStatus("error");
    }
  }

  if (status === "success") {
    return (
      <div className="flex items-center gap-3 rounded-card border border-brand-soft bg-brand-soft p-5 text-sm text-brand-deep">
        <CheckCircle2 size={20} aria-hidden="true" />
        <p>Merci, votre message a bien ete envoye. Je vous repondrai rapidement.</p>
      </div>
    );
  }

  return (
    <form
      name="contact"
      method="POST"
      onSubmit={handleSubmit}
      className="space-y-4 rounded-card border border-line bg-white p-6 shadow-card"
    >
      <div>
        <label htmlFor="contact-name" className="mb-1 block text-sm font-medium text-ink">
          Nom
        </label>
        <input
          id="contact-name"
          name="name"
          type="text"
          autoComplete="name"
          required
          value={values.name}
          onChange={(e) => update("name", e.target.value)}
          className="w-full rounded-lg border border-line px-3 py-2 text-sm text-ink focus-visible:border-brand"
        />
      </div>

      <div>
        <label htmlFor="contact-email" className="mb-1 block text-sm font-medium text-ink">
          E-mail
        </label>
        <input
          id="contact-email"
          name="email"
          type="email"
          autoComplete="email"
          required
          value={values.email}
          onChange={(e) => update("email", e.target.value)}
          className="w-full rounded-lg border border-line px-3 py-2 text-sm text-ink focus-visible:border-brand"
        />
      </div>

      <div>
        <label htmlFor="contact-message" className="mb-1 block text-sm font-medium text-ink">
          Message
        </label>
        <textarea
          id="contact-message"
          name="message"
          rows={5}
          required
          value={values.message}
          onChange={(e) => update("message", e.target.value)}
          className="w-full rounded-lg border border-line px-3 py-2 text-sm text-ink focus-visible:border-brand"
        />
      </div>

      {error && (
        <p className="flex items-center gap-2 text-sm text-rose-600">
          <AlertCircle size={16} aria-hidden="true" />
          {error}
        </p>
      )}
      {status === "error" && (
        <p className="flex items-center gap-2 text-sm text-rose-600">
          <AlertCircle size={16} aria-hidden="true" />
          L'envoi a echoue. Vous pouvez ecrire directement a{" "}
          <a href={`mailto:${PROFILE.email}`} className="underline">
            {PROFILE.email}
          </a>
          .
        </p>
      )}

      <button
        type="submit"
        disabled={status === "submitting"}
        className="inline-flex items-center gap-2 rounded-full bg-brand px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:scale-[1.02] hover:bg-brand-deep disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
      >
        <Send size={16} aria-hidden="true" />
        {status === "submitting" ? "Envoi en cours..." : "Envoyer le message"}
      </button>

      <p className="text-xs text-text-2">
        Vous pouvez aussi ecrire directement a{" "}
        <a
          href={`mailto:${PROFILE.email}`}
          className="inline-flex items-center gap-1 font-medium text-brand hover:underline"
        >
          <Mail size={12} aria-hidden="true" />
          {PROFILE.email}
        </a>
        .
      </p>
    </form>
  );
}
