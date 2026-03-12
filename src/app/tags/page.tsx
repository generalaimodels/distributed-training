import Link from "next/link";

import { getTagSummaries } from "@/lib/content";

export default function TagsPage() {
  const tags = getTagSummaries();

  return (
    <div className="page-shell">
      <section className="listing-hero">
        <span className="section-kicker">Tag archive</span>
        <h1>Topic-centric navigation for items and pages</h1>
        <p>Tags combine front matter metadata with repository structure so the site stays useful even as folders change over time.</p>
      </section>

      <div className="tag-grid">
        {tags.map((tag) => (
          <Link key={tag.slug} href={`/tags/${tag.slug}`} className="tag-card">
            <div className="eyebrow-row">
              <span>{tag.count} docs</span>
              <span>open archive</span>
            </div>
            <h2>{tag.label}</h2>
            <p>{tag.sampleDocuments.map((document) => document.title).join(" · ")}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
