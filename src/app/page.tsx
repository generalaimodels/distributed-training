import Link from "next/link";

import { DocCard } from "@/components/doc-card";
import { HomeSectionHeading } from "@/components/home-section-heading";
import { HomeSignalCard } from "@/components/home-signal-card";
import { TagPill } from "@/components/tag-pill";
import { getAllDocuments, getCollectionSummaries, getFolderSummaries, getTagSummaries } from "@/lib/content";
import { formatCompactNumber } from "@/lib/formatting";

const COVERAGE_CLUSTERS = [
  {
    title: "Structure",
    description: "Front matter, heading maps, table of contents, dividers, and section containers.",
  },
  {
    title: "Rich text",
    description: "Inline emphasis, highlights, escapes, unicode text, entities, and raw inline HTML.",
  },
  {
    title: "Code + math",
    description: "Highlighted code, terminals, diff blocks, KaTeX, matrices, aligned equations, and symbols.",
  },
  {
    title: "Evidence",
    description: "Citations, footnotes, bibliography items, theorem-style blocks, and scholarly references.",
  },
  {
    title: "Layout + forms",
    description: "Tables, responsive wrappers, forms, fieldsets, checklists, and interactive placeholders.",
  },
  {
    title: "Media",
    description: "Images, figures, audio, video, iframe embeds, SVG, diagrams, and raw HTML media tags.",
  },
];

export default function HomePage() {
  const documents = getAllDocuments();
  const collections = getCollectionSummaries();
  const folders = getFolderSummaries();
  const tags = getTagSummaries();
  const featuredDocument = documents[0] ?? null;
  const featuredDocuments = (featuredDocument ? documents.slice(1) : documents).slice(0, 6);
  const featuredTags = tags.slice(0, 14);
  const featuredCollection = collections[0] ?? null;
  const featuredFolder =
    [...folders]
      .filter((folder) => folder.depth > 0)
      .sort((left, right) => right.documentCount - left.documentCount || left.depth - right.depth)[0] ??
    folders[0] ??
    null;
  const heroMetrics = [
    {
      value: formatCompactNumber(documents.length),
      label: "documents indexed",
      detail: "Auto-discovered directly from the repository tree.",
    },
    {
      value: formatCompactNumber(folders.length),
      label: "folder archives",
      detail: "Deeply nested topic trees stay navigable instead of flattened.",
    },
    {
      value: formatCompactNumber(tags.length),
      label: "tag entry points",
      detail: "Cross-linked topic access for fast discovery and browsing.",
    },
  ];
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
  const collectionFolderLanes = collectionLanes.filter((entry) => entry.folders.length > 0).slice(0, 4);
  const featuredCollectionLanes = collectionLanes.filter((entry) => entry.documents.length > 0).slice(0, 4);

  return (
    <div className="page-shell">
      <section className="hero hero-home">
        <div className="hero-copy hero-home-copy">
          <span className="section-kicker">Advanced .md to website pipeline</span>
          <h1>Premium technical reading surfaces for every Markdown tree, notebook, and PDF library in your repository.</h1>
          <p>
            Markdown, raw HTML, equations, code, media, citations, notebooks, PDFs, and folder-aware navigation now flow into one
            polished knowledge product instead of a generic documentation dump.
          </p>
          <div className="hero-actions">
            <Link href="/docs" className="button button-primary">
              Explore documents
            </Link>
            <Link href="/folders" className="button button-secondary">
              Browse folders
            </Link>
          </div>
          <div className="hero-tag-strip" aria-label="Featured topics">
            {featuredTags.slice(0, 5).map((tag) => (
              <TagPill key={tag.slug} tag={{ slug: tag.slug, label: tag.label }} quiet />
            ))}
          </div>
          <div className="home-metric-rail">
            {heroMetrics.map((metric) => (
              <article key={metric.label} className="home-metric-card">
                <strong>{metric.value}</strong>
                <span>{metric.label}</span>
                <p>{metric.detail}</p>
              </article>
            ))}
          </div>
        </div>

        <div className="hero-bento">
          {featuredDocument ? (
            <article className="hero-feature-card hero-feature-card-primary">
              {featuredDocument.heroImage ? (
                <div className="hero-feature-media">
                  <img src={featuredDocument.heroImage} alt={featuredDocument.title} />
                </div>
              ) : null}
              <div className="hero-feature-body">
                <span className="hero-feature-kicker">Featured article</span>
                <div className="eyebrow-row">
                  <Link href={`/collections/${featuredDocument.collection.slug}`} className="eyebrow-link">
                    {featuredDocument.collection.label}
                  </Link>
                  <span>{featuredDocument.readingMinutes} min read</span>
                  <span>{featuredDocument.wordCount.toLocaleString("en-US")} words</span>
                </div>
                <h2>{featuredDocument.title}</h2>
                <p>{featuredDocument.summary}</p>
                <div className="tag-row">
                  {featuredDocument.tags.slice(0, 4).map((tag) => (
                    <TagPill key={`${featuredDocument.url}-${tag.slug}`} tag={tag} quiet />
                  ))}
                </div>
                <div className="card-actions">
                  <Link href={featuredDocument.url} className="text-link">
                    Read feature
                  </Link>
                  <Link href={featuredDocument.folderUrl} className="text-link text-link-subtle">
                    Open folder
                  </Link>
                </div>
              </div>
            </article>
          ) : null}

          {featuredCollection ? (
            <HomeSignalCard
              kicker="Lead collection"
              title={featuredCollection.label}
              description={`${featuredCollection.count} documents with a strong shared topic signal and curated reading depth.`}
              meta={featuredCollection.latestDocument ? `Latest: ${featuredCollection.latestDocument.title}` : undefined}
              href={`/collections/${featuredCollection.slug}`}
              actionLabel="Open collection"
              tone="sage"
            />
          ) : null}

          {featuredFolder ? (
            <HomeSignalCard
              kicker="Folder topology"
              title={featuredFolder.label}
              description={`${featuredFolder.documentCount} documents organized under ${featuredFolder.pathLabel}.`}
              meta={featuredFolder.latestDocument ? `Entry point: ${featuredFolder.latestDocument.title}` : undefined}
              href={featuredFolder.url}
              actionLabel="Browse folder"
              tone="accent"
            />
          ) : null}

          <article className="hero-feature-card hero-feature-card-coverage">
            <span className="hero-feature-kicker">Coverage matrix</span>
            <h2>One rendering system for structure, code, evidence, and media.</h2>
            <p>
              The site stays visually consistent while supporting generalized technical content across future subjects and folder
              hierarchies.
            </p>
            <div className="hero-coverage-list">
              {COVERAGE_CLUSTERS.map((cluster) => (
                <div key={cluster.title} className="coverage-pill">
                  <strong>{cluster.title}</strong>
                  <span>{cluster.description}</span>
                </div>
              ))}
            </div>
          </article>
        </div>
      </section>

      <section className="section-block">
        <HomeSectionHeading
          kicker="Reading room"
          title="Fresh technical documents with a stronger editorial hierarchy"
          description="The homepage now behaves like a curated landing page: fewer generic blocks, clearer emphasis, and fast access to the newest content."
          action={
            <Link href="/docs" className="text-link">
              Open all documents
            </Link>
          }
        />
        <div className="doc-grid">
          {featuredDocuments.map((document) => (
            <DocCard key={document.url} document={document} />
          ))}
        </div>
      </section>

      <section className="section-block">
        <HomeSectionHeading
          kicker="Repository architecture"
          title="Folder-first lanes preserve depth across expanding content trees"
          description="Collections surface their strongest folder branches first, so future subjects can grow without collapsing into a flat homepage."
          action={
            <Link href="/folders" className="text-link">
              Open folder index
            </Link>
          }
        />
        <div className="map-grid home-map-grid">
          {collectionFolderLanes.map(({ collection, folders: collectionFolders }) => (
            <article key={collection.slug} className="map-card">
              <div className="eyebrow-row">
                <Link href={`/collections/${collection.slug}`} className="eyebrow-link">
                  {collection.label}
                </Link>
                <span>{collection.count} docs</span>
              </div>
              <h3>{collection.label} architecture</h3>
              <p className="map-card-summary">
                Dense folder clusters and top entry points stay visible without sacrificing the premium editorial layout.
              </p>
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
        <HomeSectionHeading
          kicker="Collection lanes"
          title="Curated reading tracks grouped by subject instead of visual noise"
          description="Each lane keeps the homepage compact while still surfacing enough density to feel like a serious technical archive."
        />
        <div className="stream-grid home-stream-grid">
          {featuredCollectionLanes.map(({ collection, documents: laneDocuments }) => (
            <section key={collection.slug} className="stream-card">
              <div className="section-header section-header-compact">
                <span className="section-kicker">{collection.label}</span>
                <h2>{collection.label} reading lane</h2>
                <p className="stream-card-summary">Three fast entry points into the strongest documents for this collection.</p>
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
        <HomeSectionHeading
          kicker="Capability matrix"
          title="Generalized rendering coverage across modern technical content"
          description="The visual system stays disciplined while the content engine supports a much wider normalized schema under the hood."
        />
        <div className="coverage-grid home-coverage-grid">
          {COVERAGE_CLUSTERS.map((cluster) => (
            <article key={cluster.title} className="coverage-card coverage-card-detailed">
              <span className="coverage-dot" />
              <div className="coverage-card-copy">
                <h3>{cluster.title}</h3>
                <p>{cluster.description}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section-block">
        <HomeSectionHeading
          kicker="Tag lattice"
          title="Fast entry points into topics, items, and adjacent knowledge paths"
          description="Topic chips stay lightweight on the homepage while deeper archives remain available through the full tag index."
          action={
            <Link href="/tags" className="text-link">
              Open tag index
            </Link>
          }
        />
        <div className="tag-cloud">
          {featuredTags.map((tag) => (
            <TagPill key={tag.slug} tag={{ slug: tag.slug, label: `${tag.label} · ${tag.count}` }} />
          ))}
        </div>
      </section>
    </div>
  );
}
