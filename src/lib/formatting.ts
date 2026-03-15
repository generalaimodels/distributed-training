import type { DocumentMeta } from "@/lib/content-types";

export function formatLongDate(value: string | null): string {
  if (!value) {
    return "Living note";
  }

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "long",
  }).format(new Date(value));
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const normalized = bytes / 1024 ** exponent;
  const digits = normalized >= 10 || exponent === 0 ? 0 : 1;

  return `${normalized.toFixed(digits)} ${units[exponent]}`;
}

export function formatDocumentPrimaryStat(document: DocumentMeta): string {
  if (document.kind === "pdf") {
    return document.pageCount ? `${formatCompactNumber(document.pageCount)} pages` : "PDF reader";
  }

  return `${document.readingMinutes} min read`;
}

export function formatDocumentSecondaryStat(document: DocumentMeta): string {
  if (document.kind === "pdf") {
    return formatFileSize(document.sourceSizeBytes);
  }

  return `${document.wordCount.toLocaleString("en-US")} words`;
}
