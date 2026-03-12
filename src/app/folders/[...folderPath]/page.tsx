import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { DocCard } from "@/components/doc-card";
import { getDocumentsByFolderPath, getFolderByPath, getFolderSummaries } from "@/lib/content";

interface FolderPageProps {
  params: Promise<{
    folderPath: string[];
  }>;
}

export const dynamicParams = false;

export function generateStaticParams() {
  return getFolderSummaries().map((folder) => ({
    folderPath: folder.relativePath.split("/").filter(Boolean),
  }));
}

export async function generateMetadata({ params }: FolderPageProps): Promise<Metadata> {
  const { folderPath } = await params;
  const folder = getFolderByPath(folderPath);

  if (!folder) {
    return {};
  }

  return {
    title: `${folder.label} folder`,
    description: `${folder.documentCount} documents inside ${folder.pathLabel}.`,
  };
}

export default async function FolderPage({ params }: FolderPageProps) {
  const { folderPath } = await params;
  const folder = getFolderByPath(folderPath);

  if (!folder) {
    notFound();
  }

  const documents = getDocumentsByFolderPath(folder.relativePath);

  return (
    <div className="page-shell">
      <section className="listing-hero">
        <div className="breadcrumb-row">
          <Link href="/">Home</Link>
          <span>/</span>
          <Link href="/folders">Folders</Link>
          <span>/</span>
          <Link href={`/collections/${folder.collection.slug}`}>{folder.collection.label}</Link>
        </div>
        <span className="section-kicker">Folder archive</span>
        <h1>{folder.label}</h1>
        <p>{folder.pathLabel}</p>
        <div className="doc-meta-line">
          <span>{folder.documentCount} documents</span>
          <span>{folder.childFolders.length} child folders</span>
        </div>
        {folder.childFolders.length > 0 ? (
          <div className="tag-cloud">
            {folder.childFolders.map((childFolder) => (
              <Link key={childFolder.relativePath} href={childFolder.url} className="tag-pill">
                {childFolder.label}
              </Link>
            ))}
          </div>
        ) : null}
      </section>

      <div className="doc-grid">
        {documents.map((document) => (
          <DocCard key={document.url} document={document} />
        ))}
      </div>
    </div>
  );
}
