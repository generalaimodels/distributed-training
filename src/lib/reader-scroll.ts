const READER_GAP = 40;

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
