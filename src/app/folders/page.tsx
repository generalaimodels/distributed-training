import Link from "next/link";

import { getCollectionSummaries, getFolderSummaries } from "@/lib/content";

export default function FoldersPage() {
  const collections = getCollectionSummaries();
  const folders = getFolderSummaries();
  const collectionLanes = collections
    .map((collection) => ({
      collection,
      folders: folders.filter((folder) => folder.collection.slug === collection.slug),
    }))
    .filter((entry) => entry.folders.length > 0);

  return (
    <div className="page-shell">
      <section className="listing-hero">
        <span className="section-kicker">Folder index</span>
        <h1>All content folders, grouped by repository subject</h1>
        <p>Use this map when the repository grows deep and you want to browse by structure before opening individual documents.</p>
      </section>

      <div className="folder-index-grid">
        {collectionLanes.map(({ collection, folders: collectionFolders }) => (
          <section key={collection.slug} className="map-card">
            <div className="eyebrow-row">
              <Link href={`/collections/${collection.slug}`} className="eyebrow-link">
                {collection.label}
              </Link>
              <span>{collectionFolders.length} folders</span>
            </div>
            <h2>{collection.label}</h2>
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
          </section>
        ))}
      </div>
    </div>
  );
}
