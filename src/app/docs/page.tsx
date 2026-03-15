import Link from "next/link";

import { DocCard } from "@/components/doc-card";
import { getAllDocuments, getCollectionSummaries } from "@/lib/content";

export default function DocsPage() {
  const documents = getAllDocuments();
  const collections = getCollectionSummaries();

  return (
    <div className="page-shell">
      <section className="listing-hero">
        <span className="section-kicker">Document index</span>
        <h1>All repository pages in one structured view</h1>
        <p>The content engine reads the repository directly, so new Markdown files and PDFs show up here without restructuring the app.</p>
        <div className="tag-cloud">
          {collections.map((collection) => (
            <Link key={collection.slug} href={`/collections/${collection.slug}`} className="tag-pill">
              {collection.label}
            </Link>
          ))}
        </div>
      </section>

      <div className="doc-grid">
        {documents.map((document) => (
          <DocCard key={document.url} document={document} />
        ))}
      </div>
    </div>
  );
}
