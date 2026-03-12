import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { DocCard } from "@/components/doc-card";
import { getDocumentsByTag, getTagBySlug, getTagSummaries } from "@/lib/content";

interface TagPageProps {
  params: Promise<{
    tag: string;
  }>;
}

export const dynamicParams = false;

export function generateStaticParams() {
  return getTagSummaries().map((tag) => ({
    tag: tag.slug,
  }));
}

export async function generateMetadata({ params }: TagPageProps): Promise<Metadata> {
  const { tag } = await params;
  const currentTag = getTagBySlug(tag);

  if (!currentTag) {
    return {};
  }

  return {
    title: `${currentTag.label} tag`,
    description: `${currentTag.count} documents filed under ${currentTag.label}.`,
  };
}

export default async function TagPage({ params }: TagPageProps) {
  const { tag } = await params;
  const currentTag = getTagBySlug(tag);

  if (!currentTag) {
    notFound();
  }

  const documents = getDocumentsByTag(tag);

  return (
    <div className="page-shell">
      <section className="listing-hero">
        <span className="section-kicker">Tag archive</span>
        <h1>{currentTag.label}</h1>
        <p>{currentTag.count} documents indexed under this topic.</p>
      </section>
      <div className="doc-grid">
        {documents.map((document) => (
          <DocCard key={document.url} document={document} />
        ))}
      </div>
    </div>
  );
}
