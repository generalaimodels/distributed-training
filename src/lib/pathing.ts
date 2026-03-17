import path from "node:path";

export const CONTENT_ROOT = process.cwd();

export function toPosixPath(value: string): string {
  return value.split(path.sep).join("/");
}

export function encodePathSegments(segments: string[]): string {
  return segments.map((segment) => encodeURIComponent(segment)).join("/");
}

export function normalizeSlugSegment(value: string): string {
  const normalized = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");

  return normalized || "item";
}

export function humanizeToken(value: string): string {
  const withoutExtension = value.replace(/\.[^.]+$/, "");
  const spaced = withoutExtension.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();

  if (!spaced) {
    return "Untitled";
  }

  if (/[A-Z]/.test(spaced)) {
    return spaced;
  }

  return spaced.replace(/\b\w/g, (character) => character.toUpperCase());
}

export function isMarkdownFilename(filename: string): boolean {
  return /\.md$/i.test(filename);
}

export function isNotebookFilename(filename: string): boolean {
  return /\.ipynb$/i.test(filename);
}

export function isPdfFilename(filename: string): boolean {
  return /\.pdf$/i.test(filename);
}

export function isRenderableContentFilename(filename: string): boolean {
  return isMarkdownFilename(filename) || isNotebookFilename(filename) || isPdfFilename(filename);
}

export function docRoutePathFromSegments(segments: string[]): string {
  return `/docs/${encodePathSegments(segments)}`;
}

export function relativeFilePathToDocRouteSegments(relativeFilePath: string): string[] {
  const withoutExtension = relativeFilePath.replace(/\.[^.]+$/, "");
  let segments = toPosixPath(withoutExtension).split("/").filter(Boolean);

  if (segments.at(-1)?.toLowerCase() === "readme") {
    segments = segments.length === 1 ? ["readme"] : segments.slice(0, -1);
  }

  if (segments.length === 0) {
    segments = ["readme"];
  }

  return segments.map(normalizeSlugSegment);
}

export function relativeFilePathToDocRoutePath(relativeFilePath: string): string {
  return docRoutePathFromSegments(relativeFilePathToDocRouteSegments(relativeFilePath));
}

export function folderRoutePathFromSegments(segments: string[]): string {
  return segments.length > 0 ? `/folders/${encodePathSegments(segments)}` : "/folders";
}

export function relativeDirectoryPathToSegments(relativeDirectoryPath: string): string[] {
  return toPosixPath(relativeDirectoryPath).split("/").filter(Boolean);
}

export function relativeDirectoryPathToFolderRoutePath(relativeDirectoryPath: string): string {
  return folderRoutePathFromSegments(relativeDirectoryPathToSegments(relativeDirectoryPath));
}

export function relativeAssetPathToRoutePath(relativeAssetPath: string): string {
  const segments = toPosixPath(relativeAssetPath).split("/").filter(Boolean);
  return `/content-assets/${encodePathSegments(segments)}`;
}

export function isWithinContentRoot(targetPath: string): boolean {
  const relativePath = path.relative(CONTENT_ROOT, targetPath);
  return relativePath === "" || (!relativePath.startsWith("..") && !path.isAbsolute(relativePath));
}
