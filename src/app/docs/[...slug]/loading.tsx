export default function Loading() {
  return (
    <div className="page-shell doc-page-shell doc-loading-shell">
      <section className="doc-hero doc-hero-no-media doc-hero-loading" aria-hidden="true">
        <div className="breadcrumb-row">
          <span className="skeleton skeleton-pill skeleton-breadcrumb" />
          <span className="skeleton skeleton-pill skeleton-breadcrumb" />
          <span className="skeleton skeleton-pill skeleton-breadcrumb" />
        </div>
        <div className="doc-hero-grid doc-hero-grid-no-media">
          <div className="doc-hero-copy">
            <div className="skeleton skeleton-pill skeleton-kicker" />
            <div className="skeleton skeleton-heading-lg" />
            <div className="skeleton skeleton-heading-lg skeleton-heading-lg-short" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line skeleton-line-short" />
            <div className="doc-meta-line">
              <span className="skeleton skeleton-pill skeleton-chip" />
              <span className="skeleton skeleton-pill skeleton-chip" />
              <span className="skeleton skeleton-pill skeleton-chip" />
            </div>
          </div>
        </div>
      </section>

      <section className="doc-grid-layout">
        <article className="doc-content-panel doc-content-panel-loading" aria-hidden="true">
          <div className="content-prose">
            <div className="skeleton skeleton-heading-md" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line skeleton-line-short" />
            <div className="skeleton skeleton-block" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line skeleton-line-short" />
          </div>
        </article>
        <aside className="doc-aside" aria-hidden="true">
          <div className="aside-card">
            <div className="aside-card-header">
              <span className="skeleton skeleton-pill skeleton-chip" />
              <span className="skeleton skeleton-pill skeleton-chip-short" />
            </div>
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line skeleton-line-short" />
            <div className="skeleton skeleton-line" />
          </div>
        </aside>
      </section>
    </div>
  );
}
