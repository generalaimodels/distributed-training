export default function Loading() {
  return (
    <div className="page-shell">
      <section className="listing-hero">
        <span className="section-kicker">Loading</span>
        <h1>Preparing the reader</h1>
        <p>Resolving content structure, document links, and page metadata.</p>
      </section>
      <section className="doc-grid">
        {Array.from({ length: 6 }).map((_, index) => (
          <article key={index} className="doc-card doc-card-skeleton" aria-hidden="true">
            <div className="skeleton skeleton-line skeleton-line-short" />
            <div className="skeleton skeleton-title" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line skeleton-line-short" />
          </article>
        ))}
      </section>
    </div>
  );
}
