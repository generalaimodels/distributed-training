"use client";

import Link from "next/link";
import { useMemo } from "react";
import type { CSSProperties, ReactNode } from "react";

import { useReaderLayoutState } from "@/components/reader-layout-state";
import { ReaderToc } from "@/components/reader-toc";
import type { DocumentHeading } from "@/lib/content-types";

interface ReaderLayoutProps {
  authors: string[];
  children: ReactNode;
  collectionLabel: string;
  documentTitle: string;
  features: {
    hasMath: boolean;
    hasMermaid: boolean;
    hasRawHtml: boolean;
  };
  folderLabel: string;
  folderRelativePath: string;
  folderUrl: string;
  headings: DocumentHeading[];
  relativePath: string;
}

export function ReaderLayout({
  authors,
  children,
  collectionLabel,
  documentTitle,
  features,
  folderLabel,
  folderRelativePath,
  folderUrl,
  headings,
  relativePath,
}: ReaderLayoutProps) {
  const { focusMode } = useReaderLayoutState();

  const featureSummary = useMemo(
    () =>
      [
        features.hasRawHtml ? "raw HTML" : null,
        features.hasMath ? "math" : null,
        features.hasMermaid ? "mermaid" : null,
      ]
        .filter(Boolean)
        .join(", ") || "markdown",
    [features.hasMath, features.hasMermaid, features.hasRawHtml],
  );
  const layoutStyle = {
    "--reader-prose-measure": focusMode ? "94ch" : "88ch",
    "--reader-prose-wide-measure": focusMode ? "94ch" : "88ch",
    "--reader-sidebar-width": "332px",
  } as CSSProperties;

  return (
    <section className={focusMode ? "doc-grid-layout doc-grid-layout-focus" : "doc-grid-layout"} style={layoutStyle}>
      {children}
      <aside className={focusMode ? "doc-aside doc-aside-hidden" : "doc-aside"}>
        <ReaderToc collectionLabel={collectionLabel} documentTitle={documentTitle} headings={headings} focusMode={focusMode} />
        <div className="aside-card">
          <h2>Document data</h2>
          <dl className="meta-list">
            <div>
              <dt>Path</dt>
              <dd>{relativePath}</dd>
            </div>
            <div>
              <dt>Folder</dt>
              <dd>
                <Link href={folderUrl}>{folderRelativePath || folderLabel || "Repository root"}</Link>
              </dd>
            </div>
            <div>
              <dt>Authors</dt>
              <dd>{authors.length > 0 ? authors.join(", ") : "Repository content"}</dd>
            </div>
            <div>
              <dt>Features</dt>
              <dd>{featureSummary}</dd>
            </div>
          </dl>
        </div>
      </aside>
    </section>
  );
}
