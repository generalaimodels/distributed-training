import Link from "next/link";

export default function NotFound() {
  return (
    <div className="page-shell">
      <section className="listing-hero">
        <span className="section-kicker">Missing page</span>
        <h1>The requested page was not found.</h1>
        <p>The route may point to a file that does not exist yet, or the slug may have changed after the repository was updated.</p>
        <div className="hero-actions">
          <Link href="/docs" className="button button-primary">
            Open document index
          </Link>
          <Link href="/" className="button button-secondary">
            Return home
          </Link>
        </div>
      </section>
    </div>
  );
}
