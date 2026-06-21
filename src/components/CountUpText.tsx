import { useEffect, useRef, useState } from "react";

interface CountUpTextProps {
  text: string;
  durationMs?: number;
  className?: string;
}

const NUMBER_RE = /-?\d(?:[\d\s]*\d)?(?:[.,]\d+)?/;

function parseNumberToken(token: string): number {
  return parseFloat(token.replace(/\s/g, "").replace(",", "."));
}

function formatLike(template: string, value: number): string {
  const decimals = template.includes(",") ? template.split(",")[1]?.replace(/\D.*$/, "").length ?? 0 : 0;
  const [intPart, decPart] = value.toFixed(decimals).split(".");
  const withThousands = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  return decPart ? `${withThousands},${decPart}` : withThousands;
}

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function initialDisplay(text: string, match: RegExpMatchArray | null): string {
  if (!match || prefersReducedMotion()) return text;
  return text.replace(NUMBER_RE, formatLike(match[0], 0));
}

/** Anime le premier nombre trouvé dans `text`, en conservant tel quel le reste de la chaîne. */
export default function CountUpText({ text, durationMs = 1100, className }: CountUpTextProps) {
  const match = text.match(NUMBER_RE);
  const ref = useRef<HTMLSpanElement>(null);
  const [trackedText, setTrackedText] = useState(text);
  const [display, setDisplay] = useState(() => initialDisplay(text, match));

  // Pattern recommandé par React pour réinitialiser un état dérivé d'une prop sans effet :
  // https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes
  if (text !== trackedText) {
    setTrackedText(text);
    setDisplay(initialDisplay(text, match));
  }

  useEffect(() => {
    if (!match || prefersReducedMotion()) return;
    const token = match[0];
    const el = ref.current;
    if (!el) return;

    let started = false;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting || started) return;
        started = true;
        const target = parseNumberToken(token);
        const start = performance.now();
        function tick(now: number) {
          const progress = Math.min(1, (now - start) / durationMs);
          const eased = 1 - Math.pow(1 - progress, 3);
          setDisplay(text.replace(NUMBER_RE, formatLike(token, target * eased)));
          if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
      },
      { threshold: 0.3 }
    );
    observer.observe(el);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  );
}
