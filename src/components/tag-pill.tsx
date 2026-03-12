import Link from "next/link";

import type { TopicTag } from "@/lib/content-types";

interface TagPillProps {
  tag: TopicTag;
  href?: string | null;
  quiet?: boolean;
}

export function TagPill({ tag, href = `/tags/${tag.slug}`, quiet = false }: TagPillProps) {
  const className = quiet ? "tag-pill tag-pill-quiet" : "tag-pill";

  if (!href) {
    return <span className={className}>{tag.label}</span>;
  }

  return (
    <Link href={href} className={className}>
      {tag.label}
    </Link>
  );
}
