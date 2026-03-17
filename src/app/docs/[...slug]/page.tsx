import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { DocCard } from "@/components/doc-card";
import { MermaidLoader } from "@/components/mermaid-loader";
import { NotebookViewer } from "@/components/notebook-viewer";
import { PdfViewer } from "@/components/pdf-viewer";
import { ProseContent } from "@/components/prose-content";
import { ReaderLayout } from "@/components/reader-layout";
import { TagPill } from "@/components/tag-pill";
import { getAllDocuments, getDocumentBySlug, getMoreDocumentsFromSameFolder, getRelatedDocuments } from "@/lib/content";
import { formatDocumentPrimaryStat, formatDocumentSecondaryStat, formatLongDate } from "@/lib/formatting";

interface DocPageProps {
  params: Promise<{
    slug: string[];
  }>;
}

function normalizeComparableText(value: string): string {
  return value
    .toLowerCase()
    .replace(/&[a-z0-9#]+;/gi, " ")
    .replace(/[^a-z0-9]+/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
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
  const heroPreviewHeadings = tocHeadings.slice(0, 4);
  const firstHeading = document.headings[0] ?? null;
  const hideLeadingTitle = Boolean(
    firstHeading &&
      firstHeading.depth === 1 &&
      normalizeComparableText(firstHeading.text) === normalizeComparableText(document.title),
  );
  const heroClassName = document.heroImage ? "doc-hero" : "doc-hero doc-hero-no-media";
  const heroGridClassName = document.heroImage ? "doc-hero-grid" : "doc-hero-grid doc-hero-grid-no-media";
  const heroFeatureLabels = [
    document.kind === "pdf" ? "PDF" : null,
    document.kind === "notebook" ? "Notebook" : null,
    document.kind === "notebook" && document.notebook?.isColab ? "Colab" : null,
    document.kind === "notebook" && document.notebook?.hasOutputs ? "Outputs" : null,
    document.features.hasRawHtml ? "Raw HTML" : null,
    document.features.hasMath ? "Math" : null,
    document.features.hasMermaid ? "Mermaid" : null,
  ].filter(Boolean);
  const heroPreviewItems =
    document.kind === "pdf"
      ? ["Progressive page loading", "Smooth canvas rendering", "Outline-aware navigation", "Raw PDF access"]
      : document.kind === "notebook"
        ? [
            document.notebook?.language ? `${document.notebook.language} kernel` : null,
            document.notebook ? `${document.notebook.cellCount} notebook cells` : null,
            document.notebook?.hasOutputs ? "Saved notebook outputs" : "Source-first notebook view",
            "Load more sections on demand",
          ].filter(Boolean)
      : heroPreviewHeadings.map((heading) => heading.text);

  return (
    <div className="page-shell doc-page-shell">
      <section className={heroClassName}>
        <div className="breadcrumb-row">
          <Link href="/">Home</Link>
          <span>/</span>
          <Link href="/docs">Docs</Link>
          <span>/</span>
          <Link href={`/collections/${document.collection.slug}`}>{document.collection.label}</Link>
          <span>/</span>
          <Link href={document.folderUrl}>{document.folderLabel}</Link>
        </div>
        <div className={heroGridClassName}>
          <div className="doc-hero-copy">
            <span className="section-kicker">{document.collection.label} collection</span>
            <h1>{document.title}</h1>
            <p className="doc-summary">{document.summary}</p>
            <div className="doc-meta-line">
              <span>{formatLongDate(document.publishedAt)}</span>
              <span>{formatDocumentPrimaryStat(document)}</span>
              <span>{formatDocumentSecondaryStat(document)}</span>
            </div>
            <div className="tag-row">
              {document.tags.map((tag) => (
                <TagPill key={`${document.url}-${tag.slug}`} tag={tag} quiet />
              ))}
            </div>
          </div>
          {document.heroImage ? (
            <div className="doc-hero-side">
              <figure className="doc-hero-media-shell">
                <span className="doc-hero-side-kicker">Featured visual</span>
                <div className="doc-hero-media">
                  <img src={document.heroImage} alt={document.title} />
                </div>
                <figcaption className="doc-hero-side-meta">
                  <span>{document.collection.label}</span>
                  <span>{formatDocumentPrimaryStat(document)}</span>
                  <span>{formatDocumentSecondaryStat(document)}</span>
                </figcaption>
              </figure>
            </div>
          ) : (
            <aside className="doc-hero-side doc-hero-side-fallback" aria-label="Document overview">
              <div className="doc-hero-fallback-card">
                <span className="doc-hero-side-kicker">{document.kind === "pdf" ? "Reader brief" : "Reader brief"}</span>
                <div className="doc-hero-fallback-stats">
                  <div>
                    <strong>{formatDocumentPrimaryStat(document)}</strong>
                    <span>{document.kind === "pdf" ? "Format" : "Reading time"}</span>
                  </div>
                  <div>
                    <strong>{formatDocumentSecondaryStat(document)}</strong>
                    <span>{document.kind === "pdf" ? "File detail" : "Words"}</span>
                  </div>
                </div>
                {heroPreviewItems.length > 0 ? (
                  <div className="doc-hero-outline-preview">
                    <span className="doc-hero-outline-title">
                      {document.kind === "pdf" ? "Inside this reader" : "Inside this document"}
                    </span>
                    <ul className="doc-hero-outline-list">
                      {heroPreviewItems.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                <div className="doc-hero-side-meta doc-hero-side-meta-wrap">
                  <span>{document.folderLabel}</span>
                  {heroFeatureLabels.map((featureLabel) => (
                    <span key={featureLabel}>{featureLabel}</span>
                  ))}
                </div>
              </div>
            </aside>
          )}
        </div>
      </section>

      {document.kind === "pdf" && document.assetUrl ? (
        <PdfViewer
          assetUrl={document.assetUrl}
          collectionLabel={document.collection.label}
          documentTitle={document.title}
          folderLabel={document.folderLabel}
          relativePath={document.relativePath}
          sourceSizeBytes={document.sourceSizeBytes}
        />
      ) : document.kind === "notebook" ? (
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
          <NotebookViewer
            assetUrl={document.assetUrl}
            documentTitle={document.title}
            headingAliasMap={document.notebookHeadingAliasMap ?? {}}
            headingSectionMap={document.notebookHeadingSectionMap ?? {}}
            metrics={document.notebook}
            relativePath={document.relativePath}
            sections={document.notebookSections ?? []}
            sourceSizeBytes={document.sourceSizeBytes}
          />
        </ReaderLayout>
      ) : (
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
            <ProseContent html={document.html} hideLeadingTitle={hideLeadingTitle} />
          </div>
        </ReaderLayout>
      )}

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
