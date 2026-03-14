"use client";

import { startTransition, useEffect, useEffectEvent, useRef, useState } from "react";
import type { MouseEvent } from "react";

import type { DocumentHeading } from "@/lib/content-types";
import { getReaderOffset, scrollToAnchorId } from "@/lib/reader-scroll";
import { useReaderStateActions } from "@/components/reader-state";

interface ReaderTocProps {
  collectionLabel: string;
  documentTitle: string;
  focusMode: boolean;
  headings: DocumentHeading[];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function ReaderToc({ collectionLabel, documentTitle, focusMode, headings }: ReaderTocProps) {
  const { clearReaderState, patchReaderState, replaceReaderState } = useReaderStateActions();
  const [activeId, setActiveId] = useState<string | null>(headings[0]?.id ?? null);
  const [progress, setProgress] = useState(0);
  const tocRef = useRef<HTMLElement | null>(null);
  const frameRef = useRef<number | null>(null);
  const activeIdRef = useRef<string | null>(headings[0]?.id ?? null);
  const activeHeadingTextRef = useRef<string | null>(headings[0]?.text ?? null);
  const progressRef = useRef(0);

  const syncReadingState = useEffectEvent(() => {
    const articleElement = document.querySelector<HTMLElement>(".content-prose");

    if (!articleElement) {
      return;
    }

    const readerOffset = getReaderOffset();

    const headingElements = headings
      .map((heading) => ({
        element: document.getElementById(heading.id),
        heading,
      }))
      .filter((entry): entry is { element: HTMLElement; heading: DocumentHeading } => Boolean(entry.element));

    const articleTop = window.scrollY + articleElement.getBoundingClientRect().top - readerOffset;
    const articleBottom = articleTop + articleElement.offsetHeight - window.innerHeight * 0.4;
    const nextProgress = clamp(
      (window.scrollY - articleTop) / Math.max(articleBottom - articleTop, 1),
      0,
      1,
    );

    let nextActiveHeading = headingElements[0]?.heading ?? headings[0] ?? null;

    for (const entry of headingElements) {
      if (entry.element.getBoundingClientRect().top - readerOffset <= 18) {
        nextActiveHeading = entry.heading;
        continue;
      }

      break;
    }

    const nextActiveId = nextActiveHeading?.id ?? null;
    const nextActiveHeadingText = nextActiveHeading?.text ?? null;
    const progressChanged = Math.abs(progressRef.current - nextProgress) > 0.01;
    const headingChanged = nextActiveHeadingText !== activeHeadingTextRef.current;

    if (nextActiveId !== activeIdRef.current) {
      activeIdRef.current = nextActiveId;
      startTransition(() => {
        setActiveId(nextActiveId);
      });
    }

    if (progressChanged) {
      progressRef.current = nextProgress;
      startTransition(() => {
        setProgress(nextProgress);
      });
    }

    if (progressChanged || headingChanged) {
      activeHeadingTextRef.current = nextActiveHeadingText;
      patchReaderState({
        activeHeading: nextActiveHeadingText,
        progress: nextProgress,
      });
    }
  });

  useEffect(() => {
    const initialHeading = headings[0]?.text ?? null;

    replaceReaderState({
      collectionLabel,
      documentTitle,
      activeHeading: initialHeading,
      progress: 0,
    });

    syncReadingState();

    const scheduleSync = () => {
      if (frameRef.current !== null) {
        return;
      }

      frameRef.current = window.requestAnimationFrame(() => {
        frameRef.current = null;
        syncReadingState();
      });
    };

    window.addEventListener("scroll", scheduleSync, { passive: true });
    window.addEventListener("resize", scheduleSync);

    return () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }

      window.removeEventListener("scroll", scheduleSync);
      window.removeEventListener("resize", scheduleSync);
      clearReaderState();
    };
  }, [clearReaderState, collectionLabel, documentTitle, headings, replaceReaderState]);

  useEffect(() => {
    const syncHashTarget = () => {
      const hash = decodeURIComponent(window.location.hash.replace(/^#/, ""));

      if (!hash) {
        return;
      }

      window.requestAnimationFrame(() => {
        scrollToAnchorId(hash);
      });
    };

    syncHashTarget();
    window.addEventListener("hashchange", syncHashTarget);

    return () => {
      window.removeEventListener("hashchange", syncHashTarget);
    };
  }, [headings]);

  useEffect(() => {
    if (focusMode) {
      return;
    }

    const activeLink = tocRef.current?.querySelector<HTMLAnchorElement>(`a[data-heading-id="${activeId ?? ""}"]`);

    activeLink?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [activeId, focusMode]);

  const handleHeadingClick = (event: MouseEvent<HTMLAnchorElement>, headingId: string) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
      return;
    }

    event.preventDefault();
    scrollToAnchorId(headingId);
  };

  if (focusMode) {
    return null;
  }

  if (headings.length === 0) {
    return (
      <div className="aside-card">
        <div className="aside-card-header">
          <h2>On this page</h2>
        </div>
        <p>No generated outline for this page.</p>
      </div>
    );
  }

  return (
    <div className="aside-card">
      <div className="aside-card-header">
        <h2>On this page</h2>
        <span className="toc-progress-pill">{Math.round(progress * 100)}%</span>
      </div>
      <nav ref={tocRef} className="toc-list">
        {headings.map((heading) => {
          const active = heading.id === activeId;

          return (
            <a
              key={heading.id}
              href={`#${heading.id}`}
              data-heading-id={heading.id}
              onClick={(event) => handleHeadingClick(event, heading.id)}
              className={active ? `toc-link toc-link-active toc-depth-${heading.depth}` : `toc-link toc-depth-${heading.depth}`}
            >
              {heading.text}
            </a>
          );
        })}
      </nav>
    </div>
  );
}
