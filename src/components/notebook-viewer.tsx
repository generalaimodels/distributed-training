"use client";

import { startTransition, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";

import type { NotebookMetrics, NotebookSection } from "@/lib/content-types";
import { formatFileSize } from "@/lib/formatting";
import { ProseContent } from "@/components/prose-content";
import { READER_ANCHOR_REQUEST_EVENT, scrollToAnchorId } from "@/lib/reader-scroll";

interface NotebookViewerProps {
  assetUrl: string | null;
  documentTitle: string;
  headingAliasMap: Record<string, string>;
  headingSectionMap: Record<string, number>;
  metrics: NotebookMetrics | null;
  relativePath: string;
  sections: NotebookSection[];
  sourceSizeBytes: number;
}

const INITIAL_SECTION_BATCH = 14;
const SECTION_BATCH_SIZE = 12;
const SECTION_PREFETCH_WINDOW = 2;

function scrollWhenReady(targetId: string, attemptsLeft = 12): void {
  if (typeof document === "undefined" || typeof window === "undefined") {
    return;
  }

  if (document.getElementById(targetId)) {
    scrollToAnchorId(targetId);
    return;
  }

  if (attemptsLeft <= 0) {
    return;
  }

  window.setTimeout(() => {
    scrollWhenReady(targetId, attemptsLeft - 1);
  }, 48);
}

export function NotebookViewer({
  assetUrl,
  documentTitle,
  headingAliasMap,
  headingSectionMap,
  metrics,
  relativePath,
  sections,
  sourceSizeBytes,
}: NotebookViewerProps) {
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const [visibleSectionCount, setVisibleSectionCount] = useState(() => Math.min(INITIAL_SECTION_BATCH, sections.length));
  const visibleHtml = useMemo(
    () => sections.slice(0, visibleSectionCount).map((section) => section.html).join("\n"),
    [sections, visibleSectionCount],
  );
  const ensureSectionVisible = useEffectEvent((requestedId: string) => {
    const normalizedId = decodeURIComponent(requestedId);
    const resolvedId = headingAliasMap[normalizedId] ?? headingAliasMap[requestedId] ?? normalizedId;
    const sectionIndex = headingSectionMap[resolvedId] ?? headingSectionMap[normalizedId] ?? headingSectionMap[requestedId];

    if (typeof sectionIndex !== "number") {
      return null;
    }

    startTransition(() => {
      setVisibleSectionCount((currentValue) =>
        Math.min(sections.length, Math.max(currentValue, sectionIndex + 1 + SECTION_PREFETCH_WINDOW)),
      );
    });

    return resolvedId;
  });

  useEffect(() => {
    startTransition(() => {
      setVisibleSectionCount(Math.min(INITIAL_SECTION_BATCH, sections.length));
    });
  }, [sections.length]);

  useEffect(() => {
    const sentinelElement = sentinelRef.current;

    if (!sentinelElement || visibleSectionCount >= sections.length) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) {
          return;
        }

        startTransition(() => {
          setVisibleSectionCount((currentValue) => Math.min(sections.length, currentValue + SECTION_BATCH_SIZE));
        });
      },
      {
        rootMargin: "1200px 0px",
      },
    );

    observer.observe(sentinelElement);

    return () => {
      observer.disconnect();
    };
  }, [sections.length, visibleSectionCount]);

  useEffect(() => {
    const handleAnchorRequest = (event: Event) => {
      const requestedId =
        event instanceof CustomEvent && event.detail && typeof event.detail.id === "string" ? event.detail.id : null;

      if (!requestedId) {
        return;
      }

      const resolvedId = ensureSectionVisible(requestedId);

      if (resolvedId) {
        scrollWhenReady(resolvedId);
      }
    };

    window.addEventListener(READER_ANCHOR_REQUEST_EVENT, handleAnchorRequest);

    const hash = decodeURIComponent(window.location.hash.replace(/^#/, ""));

    if (hash) {
      const resolvedId = ensureSectionVisible(hash);

      if (resolvedId) {
        scrollWhenReady(resolvedId);
      }
    }

    return () => {
      window.removeEventListener(READER_ANCHOR_REQUEST_EVENT, handleAnchorRequest);
    };
  }, [ensureSectionVisible]);

  const handleLoadMore = () => {
    startTransition(() => {
      setVisibleSectionCount((currentValue) => Math.min(sections.length, currentValue + SECTION_BATCH_SIZE));
    });
  };

  return (
    <div className="doc-content-panel notebook-content-panel">
      <div className="notebook-toolbar">
        <div className="notebook-toolbar-copy">
          <span className="section-kicker">Notebook reader</span>
          <h2>{documentTitle}</h2>
          <p>
            Jupyter and Colab notebooks render as progressive reading surfaces, so long notebooks stay fast while code, markdown,
            and outputs remain readable.
          </p>
        </div>
        <div className="notebook-toolbar-controls">
          <div className="notebook-toolbar-stats">
            <span>{metrics ? `${metrics.cellCount} cells` : "Notebook"}</span>
            {metrics?.language ? <span>{metrics.language}</span> : null}
            {metrics?.kernelDisplayName ? <span>{metrics.kernelDisplayName}</span> : null}
            {metrics?.hasOutputs ? <span>Saved outputs</span> : <span>Source-first view</span>}
            <span>{formatFileSize(sourceSizeBytes)}</span>
          </div>
          <div className="notebook-toolbar-actions">
            <span className="notebook-progress-chip">{visibleSectionCount} / {sections.length} sections loaded</span>
            {assetUrl ? (
              <a href={assetUrl} className="pdf-control-link" target="_blank" rel="noreferrer">
                Open raw notebook
              </a>
            ) : null}
          </div>
        </div>
      </div>

      {sections.length === 0 ? (
        <div className="pdf-empty-state">
          <h3>No readable notebook cells found</h3>
          <p>This notebook does not contain saved markdown, code, or output cells that can be rendered inside the reader.</p>
        </div>
      ) : (
        <>
          <ProseContent html={visibleHtml} />
          {visibleSectionCount < sections.length ? (
            <div className="notebook-load-more">
              <button type="button" className="pdf-control-button" onClick={handleLoadMore}>
                Load more notebook sections
              </button>
              <span>{relativePath}</span>
              <div ref={sentinelRef} className="pdf-load-sentinel" aria-hidden="true" />
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
