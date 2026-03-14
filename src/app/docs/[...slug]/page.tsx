import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { DocCard } from "@/components/doc-card";
import { MermaidLoader } from "@/components/mermaid-loader";
import { ProseContent } from "@/components/prose-content";
import { ReaderLayout } from "@/components/reader-layout";
import { TagPill } from "@/components/tag-pill";
import { getAllDocuments, getDocumentBySlug, getMoreDocumentsFromSameFolder, getRelatedDocuments } from "@/lib/content";
import { formatLongDate } from "@/lib/formatting";

interface DocPageProps {
  params: Promise<{
    slug: string[];
  }>;
}

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllDocuments().map((document) => ({
    slug: document.routeSegments,
  }));
}

export async function generateMetadata({ params }: DocPageProps): Promise<Metadata> {
  const { slug } = await params;
  const document = await getDocumentBySlug(slug);

  if (!document) {
    return {};
  }

  return {
    title: document.title,
    description: document.summary,
  };
}

export default async function DocPage({ params }: DocPageProps) {
  const { slug } = await params;
  const document = await getDocumentBySlug(slug);

  if (!document) {
    notFound();
  }

  const relatedDocuments = getRelatedDocuments(document, 3);
  const sameFolderDocuments = getMoreDocumentsFromSameFolder(document, 4);
  const tocHeadings = document.headings.filter((heading) => heading.depth >= 2 && heading.depth <= 4);

  return (
    <div className="page-shell doc-page-shell">
      <section className="doc-hero">
        <div className="breadcrumb-row">
          <Link href="/">Home</Link>
          <span>/</span>
          <Link href="/docs">Docs</Link>
          <span>/</span>
          <Link href={`/collections/${document.collection.slug}`}>{document.collection.label}</Link>
          <span>/</span>
          <Link href={document.folderUrl}>{document.folderLabel}</Link>
        </div>
        <div className="doc-hero-grid">
          <div className="doc-hero-copy">
            <span className="section-kicker">{document.collection.label} collection</span>
            <h1>{document.title}</h1>
            <p className="doc-summary">{document.summary}</p>
            <div className="doc-meta-line">
              <span>{formatLongDate(document.publishedAt)}</span>
              <span>{document.readingMinutes} min read</span>
              <span>{document.wordCount.toLocaleString("en-US")} words</span>
            </div>
            <div className="tag-row">
              {document.tags.map((tag) => (
                <TagPill key={`${document.url}-${tag.slug}`} tag={tag} quiet />
              ))}
            </div>
          </div>
          {document.heroImage ? (
            <div className="doc-hero-media">
              <img src={document.heroImage} alt={document.title} />
            </div>
          ) : null}
        </div>
      </section>

      <ReaderLayout
        authors={document.authors}
        collectionLabel={document.collection.label}
        documentTitle={document.title}
        features={document.features}
        folderLabel={document.folderLabel}
        folderRelativePath={document.folderRelativePath}
        folderUrl={document.folderUrl}
        headings={tocHeadings}
        relativePath={document.relativePath}
      >
        <div className="doc-content-panel">
          {document.features.hasMermaid ? <MermaidLoader /> : null}
          <ProseContent html={document.html} />
        </div>
      </ReaderLayout>

      {sameFolderDocuments.length > 0 ? (
        <section className="section-block">
          <div className="section-header">
            <span className="section-kicker">Same folder</span>
            <h2>Continue reading in this content lane</h2>
          </div>
          <div className="doc-grid">
            {sameFolderDocuments.map((sameFolderDocument) => (
              <DocCard key={sameFolderDocument.url} document={sameFolderDocument} />
            ))}
          </div>
        </section>
      ) : null}

      {relatedDocuments.length > 0 ? (
        <section className="section-block">
          <div className="section-header">
            <span className="section-kicker">Related reading</span>
            <h2>Nearby items from the same knowledge graph</h2>
          </div>
          <div className="doc-grid">
            {relatedDocuments.map((relatedDocument) => (
              <DocCard key={relatedDocument.url} document={relatedDocument} />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
