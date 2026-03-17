const READER_GAP = 40;
export const READER_ANCHOR_REQUEST_EVENT = "distribution-training:reader-anchor-request";

export function getReaderOffset(): number {
  if (typeof document === "undefined") {
    return 136;
  }

  const siteHeader = document.querySelector<HTMLElement>(".site-header");
  return (siteHeader?.offsetHeight ?? 108) + READER_GAP;
}

export function scrollToAnchorId(id: string): boolean {
  if (typeof document === "undefined" || typeof window === "undefined") {
    return false;
  }

  const target = document.getElementById(id);

  if (!target) {
    if (window.location.hash !== `#${id}`) {
      window.history.replaceState(null, "", `#${id}`);
    }

    window.dispatchEvent(
      new CustomEvent(READER_ANCHOR_REQUEST_EVENT, {
        detail: {
          id,
        },
      }),
    );
    return false;
  }

  const nextTop = window.scrollY + target.getBoundingClientRect().top - getReaderOffset();

  window.scrollTo({
    top: Math.max(nextTop, 0),
    behavior: "smooth",
  });

  if (window.location.hash !== `#${id}`) {
    window.history.replaceState(null, "", `#${id}`);
  }

  return true;
}
