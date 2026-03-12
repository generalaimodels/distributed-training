import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { DocCard } from "@/components/doc-card";
import { getCollectionBySlug, getCollectionSummaries, getDocumentsByCollection, getFolderSummaries } from "@/lib/content";

interface CollectionPageProps {
  params: Promise<{
    collection: string;
  }>;
}

export const dynamicParams = false;

export function generateStaticParams() {
  return getCollectionSummaries().map((collection) => ({
    collection: collection.slug,
  }));
}

export async function generateMetadata({ params }: CollectionPageProps): Promise<Metadata> {
  const { collection } = await params;
  const currentCollection = getCollectionBySlug(collection);

  if (!currentCollection) {
    return {};
  }

  return {
    title: `${currentCollection.label} collection`,
    description: `${currentCollection.count} documents in the ${currentCollection.label} collection.`,
  };
}

export default async function CollectionPage({ params }: CollectionPageProps) {
  const { collection } = await params;
  const currentCollection = getCollectionBySlug(collection);

  if (!currentCollection) {
    notFound();
  }

  const documents = getDocumentsByCollection(collection);
  const folders = getFolderSummaries().filter((folder) => folder.collection.slug === collection);

  return (
    <div className="page-shell">
      <section className="listing-hero">
        <span className="section-kicker">Collection archive</span>
        <h1>{currentCollection.label}</h1>
        <p>{currentCollection.count} documents grouped from the same top-level repository subject.</p>
      </section>
      {folders.length > 0 ? (
        <section className="section-block">
          <div className="section-header">
            <span className="section-kicker">Folders</span>
            <h2>Folder map inside {currentCollection.label}</h2>
          </div>
          <div className="map-grid">
            {folders.map((folder) => (
              <Link key={folder.relativePath} href={folder.url} className="folder-row">
                <div className="folder-copy">
                  <strong>{folder.label}</strong>
                  <span>{folder.pathLabel}</span>
                </div>
                <span>{folder.documentCount} docs</span>
              </Link>
            ))}
          </div>
        </section>
      ) : null}
      <div className="doc-grid">
        {documents.map((document) => (
          <DocCard key={document.url} document={document} />
        ))}
      </div>
    </div>
  );
}
