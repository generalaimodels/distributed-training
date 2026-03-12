import Link from "next/link";

import { DocCard } from "@/components/doc-card";
import { TagPill } from "@/components/tag-pill";
import { getAllDocuments, getCollectionSummaries, getFolderSummaries, getTagSummaries } from "@/lib/content";
import { formatCompactNumber } from "@/lib/formatting";

const COVERAGE_ITEMS = [
  "headings and structured sections",
  "tables, lists, and task lists",
  "syntax-highlighted code blocks",
  "KaTeX math and equation blocks",
  "raw HTML tags and semantic containers",
  "images, audio, video, iframes, and figures",
  "callouts, citations, and footnotes",
  "future folder discovery without hardcoding topics",
];

export default function HomePage() {
  const documents = getAllDocuments();
  const collections = getCollectionSummaries();
  const folders = getFolderSummaries();
  const tags = getTagSummaries();
  const featuredDocuments = documents.slice(0, 6);
  const featuredTags = tags.slice(0, 18);
  const collectionLanes = collections
    .map((collection) => ({
      collection,
      folders: folders
        .filter((folder) => folder.collection.slug === collection.slug)
        .sort((left, right) => left.depth - right.depth || right.documentCount - left.documentCount)
        .slice(0, 6),
      documents: documents.filter((document) => document.collection.slug === collection.slug).slice(0, 3),
    }))
    .filter((entry) => entry.folders.length > 0 || entry.documents.length > 0);
  const collectionFolderLanes = collectionLanes.filter((entry) => entry.folders.length > 0);

  return (
    <div className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <span className="section-kicker">Advanced .md to website pipeline</span>
          <h1>Technical Markdown becomes a stylish, structured, future-ready knowledge site.</h1>
          <p>
            Every Markdown file in the repository is discovered automatically and rendered with support for front matter, raw HTML,
            equations, code, tables, media, citations, callouts, and folder growth across new subjects.
          </p>
          <div className="hero-actions">
            <Link href="/docs" className="button button-primary">
              Explore documents
            </Link>
            <Link href="/folders" className="button button-secondary">
              Browse folders
            </Link>
          </div>
        </div>
        <div className="hero-panel">
          <div className="metric-card">
            <span className="metric-value">{formatCompactNumber(documents.length)}</span>
            <span className="metric-label">documents indexed</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{formatCompactNumber(folders.length)}</span>
            <span className="metric-label">content folders</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{formatCompactNumber(tags.length)}</span>
            <span className="metric-label">tag archives</span>
          </div>
        </div>
      </section>

      <section className="section-block">
        <div className="section-header">
          <span className="section-kicker">Renderer coverage</span>
          <h2>Normalized support for modern technical content</h2>
        </div>
        <div className="coverage-grid">
          {COVERAGE_ITEMS.map((item) => (
            <article key={item} className="coverage-card">
              <span className="coverage-dot" />
              <p>{item}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section-block">
        <div className="section-header">
          <span className="section-kicker">Repository map</span>
          <h2>Folder-first navigation for large content trees</h2>
          <p>The main page now uses real folder paths so deep content stays organized instead of being flattened into a few top-level cards.</p>
        </div>
        <div className="map-grid">
          {collectionFolderLanes.map(({ collection, folders: collectionFolders }) => (
            <article key={collection.slug} className="map-card">
              <div className="eyebrow-row">
                <Link href={`/collections/${collection.slug}`} className="eyebrow-link">
                  {collection.label}
                </Link>
                <span>{collection.count} docs</span>
              </div>
              <h3>{collection.label}</h3>
              <div className="folder-list">
                {collectionFolders.map((folder) => (
                  <Link key={folder.relativePath} href={folder.url} className="folder-row">
                    <div className="folder-copy">
                      <strong>{folder.label}</strong>
                      <span>{folder.pathLabel}</span>
                    </div>
                    <span>{folder.documentCount} docs</span>
                  </Link>
                ))}
              </div>
              <div className="tag-row">
                {collection.topTags.map((tag) => (
                  <TagPill key={`${collection.slug}-${tag.slug}`} tag={tag} quiet />
                ))}
              </div>
              <div className="card-actions">
                <Link href={`/collections/${collection.slug}`} className="text-link">
                  Open collection
                </Link>
                {collection.latestDocument ? (
                  <Link href={collection.latestDocument.url} className="text-link text-link-subtle">
                    Read latest
                  </Link>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section-block">
        <div className="section-header">
          <span className="section-kicker">Reading lanes</span>
          <h2>Recent content arranged by collection</h2>
        </div>
        <div className="stream-grid">
          {collectionLanes.map(({ collection, documents: laneDocuments }) => (
            <section key={collection.slug} className="stream-card">
              <div className="section-header section-header-compact">
                <span className="section-kicker">{collection.label}</span>
                <h2>{collection.label} reading lane</h2>
              </div>
              <div className="doc-grid doc-grid-compact">
                {laneDocuments.map((document) => (
                  <DocCard key={document.url} document={document} />
                ))}
              </div>
              <div className="card-actions">
                <Link href={`/collections/${collection.slug}`} className="text-link">
                  View all in {collection.label}
                </Link>
              </div>
            </section>
          ))}
        </div>
      </section>

      <section className="section-block">
        <div className="section-header">
          <span className="section-kicker">Latest pages</span>
          <h2>Fast access to newly indexed documents</h2>
        </div>
        <div className="doc-grid">
          {featuredDocuments.map((document) => (
            <DocCard key={document.url} document={document} />
          ))}
        </div>
      </section>

      <section className="section-block">
        <div className="section-header">
          <span className="section-kicker">Tag lattice</span>
          <h2>Fast entry points into items and topics</h2>
        </div>
        <div className="tag-cloud">
          {featuredTags.map((tag) => (
            <TagPill key={tag.slug} tag={{ slug: tag.slug, label: `${tag.label} · ${tag.count}` }} />
          ))}
        </div>
      </section>
    </div>
  );
}
