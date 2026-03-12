import Link from "next/link";

import { formatLongDate } from "@/lib/formatting";
import type { DocumentMeta } from "@/lib/content-types";
import { TagPill } from "@/components/tag-pill";

interface DocCardProps {
  document: DocumentMeta;
}

export function DocCard({ document }: DocCardProps) {
  return (
    <article className="doc-card">
      <div className="eyebrow-row">
        <Link href={`/collections/${document.collection.slug}`} className="eyebrow-link" scroll>
          {document.collection.label}
        </Link>
        <Link href={document.folderUrl} className="eyebrow-subtle-link" scroll>
          {document.folderLabel}
        </Link>
        <span>{formatLongDate(document.publishedAt)}</span>
      </div>
      <Link href={document.url} className="doc-card-link" scroll>
        <h3>{document.title}</h3>
      </Link>
      <p className="doc-card-summary">{document.summary}</p>
      <div className="doc-card-meta">
        <span>{document.readingMinutes} min read</span>
        <span>{document.wordCount.toLocaleString("en-US")} words</span>
      </div>
      <div className="tag-row">
        {document.tags.slice(0, 4).map((tag) => (
          <TagPill key={`${document.url}-${tag.slug}`} tag={tag} quiet />
        ))}
      </div>
      <div className="card-actions">
        <Link href={document.url} className="text-link" scroll>
          Read article
        </Link>
        <Link href={document.folderUrl} className="text-link text-link-subtle" scroll>
          Open folder
        </Link>
      </div>
    </article>
  );
}
