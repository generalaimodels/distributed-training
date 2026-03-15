"use client";

import { startTransition, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import type { MouseEvent } from "react";

import { useReaderLayoutState } from "@/components/reader-layout-state";
import { useReaderStateActions } from "@/components/reader-state";
import { formatFileSize } from "@/lib/formatting";
import { getReaderOffset } from "@/lib/reader-scroll";

type PdfJsModule = typeof import("pdfjs-dist");
type PdfDocumentProxy = import("pdfjs-dist/types/src/display/api").PDFDocumentProxy;
type PdfOutlineSource = {
  dest: string | Array<unknown> | null;
  items: PdfOutlineSource[];
  title: string;
  url: string | null;
};

interface PdfOutlineEntry {
  depth: number;
  id: string;
  pageNumber: number;
  title: string;
}

interface PdfViewerProps {
  assetUrl: string;
  collectionLabel: string;
  documentTitle: string;
  folderLabel: string;
  relativePath: string;
  sourceSizeBytes: number;
}

interface PdfPageCanvasProps {
  fallbackAspectRatio: number;
  pageNumber: number;
  pdfDocument: PdfDocumentProxy;
  targetWidth: number;
  zoom: number;
}

const INITIAL_PAGE_BATCH = 6;
const PAGE_BATCH_SIZE = 8;
const PAGE_PREFETCH_WINDOW = 2;

let pdfJsModulePromise: Promise<PdfJsModule> | null = null;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function buildPageJumpEntries(pageCount: number): PdfOutlineEntry[] {
  const step = pageCount <= 20 ? 1 : pageCount <= 80 ? 5 : 10;
  const entries: PdfOutlineEntry[] = [];

  for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += step) {
    entries.push({
      depth: 2,
      id: `page-${pageNumber}`,
      pageNumber,
      title: `Page ${pageNumber}`,
    });
  }

  if (entries.at(-1)?.pageNumber !== pageCount) {
    entries.push({
      depth: 2,
      id: `page-${pageCount}`,
      pageNumber: pageCount,
      title: `Page ${pageCount}`,
    });
  }

  return entries;
}

function normalizeOutlineTitle(value: string): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized || "Untitled section";
}

function createOutlineId(title: string, pageNumber: number, depth: number): string {
  const normalizedTitle = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return `${normalizedTitle || "section"}-${depth}-${pageNumber}`;
}

async function loadPdfJsModule(): Promise<PdfJsModule> {
  if (!pdfJsModulePromise) {
    pdfJsModulePromise = import("pdfjs-dist").then((module) => {
      module.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
      return module;
    });
  }

  return pdfJsModulePromise;
}

async function resolveDestinationPageNumber(
  pdfDocument: PdfDocumentProxy,
  destination: string | Array<unknown> | null,
): Promise<number | null> {
  if (!destination) {
    return null;
  }

  const resolvedDestination = typeof destination === "string" ? await pdfDocument.getDestination(destination) : destination;

  if (!resolvedDestination || resolvedDestination.length === 0) {
    return null;
  }

  const target = resolvedDestination[0];

  if (typeof target === "number") {
    return target + 1;
  }

  try {
    const pageIndex = await pdfDocument.getPageIndex(target as Parameters<PdfDocumentProxy["getPageIndex"]>[0]);
    return pageIndex + 1;
  } catch {
    return null;
  }
}

async function flattenOutlineEntries(
  pdfDocument: PdfDocumentProxy,
  sourceItems: PdfOutlineSource[],
  depth = 2,
): Promise<PdfOutlineEntry[]> {
  const entries: PdfOutlineEntry[] = [];

  for (const item of sourceItems) {
    if (!item.url) {
      const pageNumber = await resolveDestinationPageNumber(pdfDocument, item.dest);

      if (pageNumber) {
        const title = normalizeOutlineTitle(item.title);
        entries.push({
          depth,
          id: createOutlineId(title, pageNumber, depth),
          pageNumber,
          title,
        });
      }
    }

    if (item.items.length > 0) {
      entries.push(...(await flattenOutlineEntries(pdfDocument, item.items, depth + 1)));
    }
  }

  return entries;
}

function scrollToPdfPage(pageNumber: number): void {
  const pageElement = document.getElementById(`pdf-page-${pageNumber}`);

  if (!pageElement) {
    return;
  }

  const targetTop = window.scrollY + pageElement.getBoundingClientRect().top - getReaderOffset() - 18;
  window.scrollTo({
    top: Math.max(0, targetTop),
    behavior: "smooth",
  });
}

function PdfPageCanvas({ fallbackAspectRatio, pageNumber, pdfDocument, targetWidth, zoom }: PdfPageCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [shouldRender, setShouldRender] = useState(pageNumber <= 2);
  const [isRendering, setIsRendering] = useState(true);
  const [renderedHeight, setRenderedHeight] = useState<number | null>(null);

  useEffect(() => {
    const containerElement = containerRef.current;

    if (!containerElement) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          startTransition(() => {
            setShouldRender(true);
          });
          observer.disconnect();
        }
      },
      {
        rootMargin: "900px 0px",
      },
    );

    observer.observe(containerElement);

    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!shouldRender || !targetWidth || !canvasRef.current) {
      return;
    }

    let cancelled = false;
    let renderTask: { cancel: () => void; promise: Promise<unknown> } | null = null;

    const renderPage = async () => {
      setIsRendering(true);

      try {
        const page = await pdfDocument.getPage(pageNumber);

        if (cancelled || !canvasRef.current) {
          return;
        }

        const baseViewport = page.getViewport({ scale: 1 });
        const scale = clamp((targetWidth / Math.max(baseViewport.width, 1)) * zoom, 0.25, 5);
        const viewport = page.getViewport({ scale });
        const outputScale = clamp(window.devicePixelRatio || 1, 1, 2);
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d", {
          alpha: false,
        });

        if (!context) {
          return;
        }

        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;

        startTransition(() => {
          setRenderedHeight(viewport.height);
        });

        renderTask = page.render({
          canvas,
          canvasContext: context,
          viewport,
          transform: outputScale === 1 ? undefined : [outputScale, 0, 0, outputScale, 0, 0],
        });

        await renderTask.promise;

        if (!cancelled) {
          startTransition(() => {
            setIsRendering(false);
          });
        }
      } catch {
        if (!cancelled) {
          startTransition(() => {
            setIsRendering(false);
          });
        }
      }
    };

    void renderPage();

    return () => {
      cancelled = true;

      try {
        renderTask?.cancel();
      } catch {
        // Ignore cancellation errors during fast navigation and zoom changes.
      }
    };
  }, [pageNumber, pdfDocument, shouldRender, targetWidth, zoom]);

  return (
    <section
      ref={containerRef}
      id={`pdf-page-${pageNumber}`}
      className="pdf-page-shell"
      data-pdf-page-number={pageNumber}
      style={{
        minHeight: renderedHeight ?? (targetWidth ? targetWidth * fallbackAspectRatio * zoom : undefined),
      }}
    >
      <div className="pdf-page-header">
        <span className="pdf-page-chip">Page {pageNumber}</span>
      </div>
      <div className={isRendering ? "pdf-page-canvas-shell pdf-page-canvas-shell-loading" : "pdf-page-canvas-shell"}>
        <canvas ref={canvasRef} />
        {isRendering ? <span className="pdf-page-rendering">Rendering page...</span> : null}
      </div>
    </section>
  );
}

export function PdfViewer({
  assetUrl,
  collectionLabel,
  documentTitle,
  folderLabel,
  relativePath,
  sourceSizeBytes,
}: PdfViewerProps) {
  const { focusMode } = useReaderLayoutState();
  const { clearReaderState, patchReaderState, replaceReaderState } = useReaderStateActions();
  const contentRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const activePageRef = useRef(1);
  const progressRef = useRef(0);
  const [pdfDocument, setPdfDocument] = useState<PdfDocumentProxy | null>(null);
  const [outlineEntries, setOutlineEntries] = useState<PdfOutlineEntry[]>([]);
  const [loadedPagesCount, setLoadedPagesCount] = useState(0);
  const [pageCount, setPageCount] = useState(0);
  const [activePage, setActivePage] = useState(1);
  const [containerWidth, setContainerWidth] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [defaultAspectRatio, setDefaultAspectRatio] = useState(1.33);
  const [error, setError] = useState<string | null>(null);
  const outlineEntriesForDisplay = useMemo(
    () => (outlineEntries.length > 0 ? outlineEntries : buildPageJumpEntries(pageCount)),
    [outlineEntries, pageCount],
  );
  const visiblePageNumbers = useMemo(
    () => Array.from({ length: loadedPagesCount }, (_, index) => index + 1),
    [loadedPagesCount],
  );

  const syncReaderState = useEffectEvent(() => {
    const contentElement = contentRef.current;

    if (!contentElement || pageCount === 0) {
      return;
    }

    const pageElements = Array.from(contentElement.querySelectorAll<HTMLElement>("[data-pdf-page-number]"));

    if (pageElements.length === 0) {
      return;
    }

    const readerOffset = getReaderOffset();
    let nextActivePage = 1;

    for (const pageElement of pageElements) {
      const pageNumber = Number(pageElement.dataset.pdfPageNumber ?? "0");

      if (pageNumber > 0 && pageElement.getBoundingClientRect().top - readerOffset <= 20) {
        nextActivePage = pageNumber;
        continue;
      }

      break;
    }

    const nextProgress = pageCount > 1 ? clamp((nextActivePage - 1) / (pageCount - 1), 0, 1) : 0;

    const activeOutlineEntry =
      [...outlineEntriesForDisplay]
        .reverse()
        .find((entry) => entry.pageNumber <= nextActivePage) ?? null;
    const activeHeading = activeOutlineEntry?.title ?? `Page ${nextActivePage}`;
    const progressChanged = Math.abs(progressRef.current - nextProgress) > 0.01;
    const pageChanged = nextActivePage !== activePageRef.current;

    if (pageChanged) {
      activePageRef.current = nextActivePage;
      startTransition(() => {
        setActivePage(nextActivePage);
      });
    }

    if (progressChanged) {
      progressRef.current = nextProgress;
    }

    if (progressChanged || pageChanged) {
      patchReaderState({
        activeHeading,
        progress: nextProgress,
      });
    }
  });

  useEffect(() => {
    let disposed = false;
    let loadingTask: { destroy: () => void; promise: Promise<PdfDocumentProxy> } | null = null;
    let activeDocument: PdfDocumentProxy | null = null;

    replaceReaderState({
      collectionLabel,
      documentTitle,
      activeHeading: "Opening PDF",
      progress: 0,
    });

    const loadDocument = async () => {
      try {
        const pdfJs = await loadPdfJsModule();

        const nextLoadingTask = pdfJs.getDocument({
          url: assetUrl,
          disableAutoFetch: false,
          disableStream: false,
          isEvalSupported: false,
          useWorkerFetch: true,
        });
        loadingTask = nextLoadingTask;

        const loadedDocument = await nextLoadingTask.promise;

        if (disposed) {
          void loadedDocument.destroy();
          return;
        }

        activeDocument = loadedDocument;
        const firstPage = await loadedDocument.getPage(1);
        const firstViewport = firstPage.getViewport({ scale: 1 });
        const rawOutline = (await loadedDocument.getOutline()) as PdfOutlineSource[] | null;
        const flattenedOutline = rawOutline ? await flattenOutlineEntries(loadedDocument, rawOutline) : [];

        if (disposed) {
          return;
        }

        startTransition(() => {
          setPdfDocument(loadedDocument);
          setPageCount(loadedDocument.numPages);
          setLoadedPagesCount(Math.min(INITIAL_PAGE_BATCH, loadedDocument.numPages));
          setOutlineEntries(flattenedOutline);
          setDefaultAspectRatio(firstViewport.height / Math.max(firstViewport.width, 1));
          setError(null);
        });
      } catch (loadError) {
        if (disposed) {
          return;
        }

        const message = loadError instanceof Error ? loadError.message : "Unable to load this PDF.";

        startTransition(() => {
          setError(message);
          setPdfDocument(null);
          setPageCount(0);
          setLoadedPagesCount(0);
          setOutlineEntries([]);
        });
      }
    };

    void loadDocument();

    return () => {
      disposed = true;
      clearReaderState();

      try {
        loadingTask?.destroy();
      } catch {
        // Ignore task disposal failures during route transitions.
      }

      if (activeDocument) {
        void activeDocument.destroy();
      }
    };
  }, [assetUrl, clearReaderState, collectionLabel, documentTitle, replaceReaderState]);

  useEffect(() => {
    const contentElement = contentRef.current;

    if (!contentElement) {
      return;
    }

    const updateWidth = () => {
      startTransition(() => {
        setContainerWidth(Math.max(contentElement.clientWidth - 48, 0));
      });
    };

    updateWidth();

    const observer = new ResizeObserver(() => {
      updateWidth();
    });

    observer.observe(contentElement);
    window.addEventListener("resize", updateWidth);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateWidth);
    };
  }, []);

  useEffect(() => {
    const sentinelElement = sentinelRef.current;

    if (!sentinelElement || pageCount === 0 || loadedPagesCount >= pageCount) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) {
          return;
        }

        startTransition(() => {
          setLoadedPagesCount((currentValue) => Math.min(pageCount, currentValue + PAGE_BATCH_SIZE));
        });
      },
      {
        rootMargin: "900px 0px",
      },
    );

    observer.observe(sentinelElement);

    return () => {
      observer.disconnect();
    };
  }, [loadedPagesCount, pageCount]);

  useEffect(() => {
    const scheduleSync = () => {
      window.requestAnimationFrame(() => {
        syncReaderState();
      });
    };

    scheduleSync();
    window.addEventListener("scroll", scheduleSync, { passive: true });
    window.addEventListener("resize", scheduleSync);

    return () => {
      window.removeEventListener("scroll", scheduleSync);
      window.removeEventListener("resize", scheduleSync);
    };
  }, [syncReaderState]);

  const handleOutlineClick = (event: MouseEvent<HTMLAnchorElement>, pageNumber: number) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
      return;
    }

    event.preventDefault();

    startTransition(() => {
      setLoadedPagesCount((currentValue) => Math.min(pageCount, Math.max(currentValue, pageNumber + PAGE_PREFETCH_WINDOW)));
    });

    window.requestAnimationFrame(() => {
      window.setTimeout(() => {
        scrollToPdfPage(pageNumber);
      }, 40);
    });
  };

  const loadMorePages = () => {
    startTransition(() => {
      setLoadedPagesCount((currentValue) => Math.min(pageCount, currentValue + PAGE_BATCH_SIZE));
    });
  };

  return (
    <section className={focusMode ? "doc-grid-layout doc-grid-layout-focus" : "doc-grid-layout"}>
      <div className="doc-content-panel pdf-content-panel">
        <div className="pdf-toolbar">
          <div className="pdf-toolbar-copy">
            <span className="section-kicker">PDF reader</span>
            <h2>{documentTitle}</h2>
            <p>
              Progressive page rendering keeps long papers, books, and technical reports readable without forcing the entire PDF to
              paint at once.
            </p>
          </div>
          <div className="pdf-toolbar-controls">
            <div className="pdf-toolbar-stats">
              <span>{pageCount > 0 ? `${pageCount} pages` : "Loading pages..."}</span>
              <span>{formatFileSize(sourceSizeBytes)}</span>
              <span>{loadedPagesCount > 0 ? `${loadedPagesCount} loaded` : "Streaming on demand"}</span>
            </div>
            <div className="pdf-toolbar-actions">
              <button type="button" className="pdf-control-button" onClick={() => setZoom((value) => clamp(value - 0.1, 0.8, 1.6))}>
                A-
              </button>
              <button type="button" className="pdf-control-button" onClick={() => setZoom(1)}>
                Reset
              </button>
              <button type="button" className="pdf-control-button" onClick={() => setZoom((value) => clamp(value + 0.1, 0.8, 1.6))}>
                A+
              </button>
              <button
                type="button"
                className="pdf-control-button"
                onClick={loadMorePages}
                disabled={pageCount === 0 || loadedPagesCount >= pageCount}
              >
                Load more pages
              </button>
              <a href={assetUrl} className="pdf-control-link" target="_blank" rel="noreferrer">
                Open raw PDF
              </a>
            </div>
          </div>
        </div>

        {error ? (
          <div className="pdf-empty-state">
            <h3>Unable to load this PDF inside the reader</h3>
            <p>{error}</p>
            <a href={assetUrl} className="button button-secondary" target="_blank" rel="noreferrer">
              Open raw PDF
            </a>
          </div>
        ) : (
          <div ref={contentRef} className="pdf-page-stack">
            {pdfDocument && containerWidth > 0
              ? visiblePageNumbers.map((pageNumber) => (
                  <PdfPageCanvas
                    key={pageNumber}
                    fallbackAspectRatio={defaultAspectRatio}
                    pageNumber={pageNumber}
                    pdfDocument={pdfDocument}
                    targetWidth={containerWidth}
                    zoom={zoom}
                  />
                ))
              : (
                <div className="pdf-loading-state">
                  <div className="pdf-loading-card" />
                  <div className="pdf-loading-card pdf-loading-card-short" />
                </div>
              )}
            {loadedPagesCount < pageCount ? <div ref={sentinelRef} className="pdf-load-sentinel" aria-hidden="true" /> : null}
          </div>
        )}
      </div>

      <aside className={focusMode ? "doc-aside doc-aside-hidden" : "doc-aside"}>
        <div className="aside-card">
          <div className="aside-card-header">
            <h2>On this PDF</h2>
            <span className="toc-progress-pill">
              {pageCount > 0 ? `Page ${Math.min(activePage, pageCount)}` : "PDF"}
            </span>
          </div>
          <nav className="toc-list">
            {outlineEntriesForDisplay.map((entry) => {
              const active = activePage === entry.pageNumber;

              return (
                <a
                  key={entry.id}
                  href={`#pdf-page-${entry.pageNumber}`}
                  className={active ? "toc-link toc-link-active" : "toc-link"}
                  onClick={(event) => handleOutlineClick(event, entry.pageNumber)}
                  style={{
                    paddingLeft: `${0.68 + Math.max(entry.depth - 2, 0) * 0.78}rem`,
                  }}
                >
                  {entry.title}
                </a>
              );
            })}
          </nav>
        </div>

        <div className="aside-card">
          <h2>Reader data</h2>
          <dl className="meta-list">
            <div>
              <dt>Path</dt>
              <dd>{relativePath}</dd>
            </div>
            <div>
              <dt>Folder</dt>
              <dd>{folderLabel}</dd>
            </div>
            <div>
              <dt>File size</dt>
              <dd>{formatFileSize(sourceSizeBytes)}</dd>
            </div>
            <div>
              <dt>Loading model</dt>
              <dd>On-demand page rendering with incremental loading.</dd>
            </div>
          </dl>
        </div>
      </aside>
    </section>
  );
}
