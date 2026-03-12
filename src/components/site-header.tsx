"use client";

import { useEffect, useRef } from "react";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useReaderState } from "@/components/reader-state";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/docs", label: "Docs" },
  { href: "/folders", label: "Folders" },
  { href: "/tags", label: "Tags" },
];

function isActiveRoute(pathname: string, href: string): boolean {
  if (href === "/") {
    return pathname === "/";
  }

  return pathname === href || pathname.startsWith(`${href}/`);
}

export function SiteHeader() {
  const pathname = usePathname();
  const readerState = useReaderState();
  const headerRef = useRef<HTMLElement | null>(null);
  const isDocRoute = pathname.startsWith("/docs/");
  const readingState = isDocRoute ? readerState : null;
  const progressValue = Math.round((readingState?.progress ?? 0) * 100);

  useEffect(() => {
    const headerElement = headerRef.current;

    if (!headerElement) {
      return;
    }

    const updateHeaderHeight = () => {
      document.documentElement.style.setProperty("--site-header-height", `${Math.round(headerElement.offsetHeight)}px`);
    };

    updateHeaderHeight();

    const observer = new ResizeObserver(() => {
      updateHeaderHeight();
    });

    observer.observe(headerElement);
    window.addEventListener("resize", updateHeaderHeight);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateHeaderHeight);
    };
  }, [readingState]);

  return (
    <header ref={headerRef} className={readingState ? "site-header site-header-reading" : "site-header"}>
      <div className="container header-shell">
        <div className="header-inner">
          <Link href="/" className="brand">
            <span className="brand-kicker">Markdown intelligence</span>
            <span className="brand-title">Distribution Training</span>
          </Link>
          <nav className="site-nav" aria-label="Primary">
            {NAV_ITEMS.map((item) => {
              const active = isActiveRoute(pathname, item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={active ? "site-nav-link site-nav-link-active" : "site-nav-link"}
                  aria-current={active ? "page" : undefined}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        {readingState ? (
          <div className="header-reading-bar" aria-live="polite">
            <div className="header-reading-copy">
              <span className="header-reading-chip">{readingState.collectionLabel}</span>
              <div className="header-reading-text">
                <strong className="header-reading-title">{readingState.documentTitle}</strong>
                <span className="header-reading-section">{readingState.activeHeading ?? "Reading overview"}</span>
              </div>
            </div>
            <div className="header-reading-progress">
              <span className="header-reading-progress-label">{progressValue}% read</span>
              <div className="header-progress-track" aria-hidden="true">
                <span className="header-progress-value" style={{ transform: `scaleX(${readingState.progress})` }} />
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </header>
  );
}
