import Link from "next/link";

type SignalTone = "accent" | "sage" | "neutral";

interface HomeSignalCardProps {
  kicker: string;
  title: string;
  description: string;
  href: string;
  actionLabel: string;
  meta?: string;
  tone?: SignalTone;
}

export function HomeSignalCard({
  kicker,
  title,
  description,
  href,
  actionLabel,
  meta,
  tone = "neutral",
}: HomeSignalCardProps) {
  return (
    <article className={`hero-feature-card signal-card signal-card-${tone}`}>
      <span className="signal-card-kicker">{kicker}</span>
      <h3>{title}</h3>
      {meta ? <p className="signal-card-meta">{meta}</p> : null}
      <p>{description}</p>
      <div className="card-actions">
        <Link href={href} className="text-link">
          {actionLabel}
        </Link>
      </div>
    </article>
  );
}
