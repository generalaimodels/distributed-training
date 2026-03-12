import fs from "node:fs";
import path from "node:path";

import { parse as parseToml } from "smol-toml";
import YAML from "yaml";

import type {
  CollectionSummary,
  DocumentMeta,
  DocumentPageData,
  DocumentReference,
  FeatureFlags,
  FolderReference,
  FolderSummary,
  TagSummary,
  TopicTag,
} from "@/lib/content-types";
import { renderMarkdownDocument } from "@/lib/markdown";
import {
  CONTENT_ROOT,
  humanizeToken,
  isMarkdownFilename,
  isWithinContentRoot,
  normalizeSlugSegment,
  relativeAssetPathToRoutePath,
  relativeDirectoryPathToFolderRoutePath,
  relativeFilePathToDocRoutePath,
  relativeFilePathToDocRouteSegments,
  toPosixPath,
} from "@/lib/pathing";

const DEVELOPMENT_MODE = process.env.NODE_ENV !== "production";
const DEVELOPMENT_CACHE_TTL_MS = 250;
const IGNORED_DIRECTORIES = new Set([
  ".git",
  ".next",
  "node_modules",
  "out",
  "dist",
  "build",
  "coverage",
]);
const STRUCTURAL_SEGMENTS = new Set(["_posts", "assets", "media", "images", "figures", "blogs", "notes"]);
const LANGUAGE_SEGMENTS = new Set(["japanese", "chinese", "english"]);

let repositoryStateCache: RepositoryState | null = null;
let lastRepositoryStateCheckAt = 0;
const documentRecordCache = new Map<string, CachedDocumentRecord>();
const documentPageCache = new Map<string, CachedDocumentPage>();

interface ParsedSource {
  frontMatter: Record<string, unknown>;
  content: string;
}

interface MarkdownFileDescriptor {
  filePath: string;
  relativePath: string;
  size: number;
  mtimeMs: number;
  versionKey: string;
}

interface CachedDocumentRecord {
  versionKey: string;
  document: DocumentMeta;
}

interface CachedDocumentPage {
  versionKey: string;
  page: Promise<DocumentPageData | null>;
}

interface RepositoryState {
  snapshotKey: string;
  documents: DocumentMeta[];
  documentByRouteKey: Map<string, DocumentMeta>;
  fileVersionByRelativePath: Map<string, string>;
  documentsByCollection: Map<string, DocumentMeta[]>;
  documentsByFolder: Map<string, DocumentMeta[]>;
  documentsByTag: Map<string, DocumentMeta[]>;
  collections: CollectionSummary[];
  collectionBySlug: Map<string, CollectionSummary>;
  folders: FolderSummary[];
  folderByPath: Map<string, FolderSummary>;
  tags: TagSummary[];
  tagBySlug: Map<string, TagSummary>;
}

function walkMarkdownFiles(directory: string, results: MarkdownFileDescriptor[]): void {
  const entries = fs
    .readdirSync(directory, { withFileTypes: true })
    .sort((left, right) => left.name.localeCompare(right.name));

  for (const entry of entries) {
    const absolutePath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      if (!IGNORED_DIRECTORIES.has(entry.name)) {
        walkMarkdownFiles(absolutePath, results);
      }

      continue;
    }

    if (!entry.isFile() || !isMarkdownFilename(entry.name)) {
      continue;
    }

    const stat = fs.statSync(absolutePath);
    const relativePath = toPosixPath(path.relative(CONTENT_ROOT, absolutePath));
    const versionKey = `${relativePath}:${stat.size}:${stat.mtimeMs}`;

    results.push({
      filePath: absolutePath,
      relativePath,
      size: stat.size,
      mtimeMs: stat.mtimeMs,
      versionKey,
    });
  }
}

function listMarkdownFiles(): MarkdownFileDescriptor[] {
  const files: MarkdownFileDescriptor[] = [];
  walkMarkdownFiles(CONTENT_ROOT, files);
  return files;
}

function buildSnapshotKey(files: MarkdownFileDescriptor[]): string {
  return files.map((file) => file.versionKey).join("|");
}

function parseDelimitedFrontMatter(
  source: string,
  expression: RegExp,
  parser: (value: string) => unknown,
): ParsedSource | null {
  const match = source.match(expression);

  if (!match) {
    return null;
  }

  try {
    const parsed = parser(match[1]);

    return {
      frontMatter: isPlainObject(parsed) ? parsed : {},
      content: source.slice(match[0].length),
    };
  } catch {
    return {
      frontMatter: {},
      content: source,
    };
  }
}

function parseJsonFrontMatter(value: string): Record<string, unknown> {
  const trimmed = value.trim();

  if (!trimmed) {
    return {};
  }

  const parsed = JSON.parse(trimmed) as unknown;
  return isPlainObject(parsed) ? parsed : {};
}

function parseSource(rawSource: string): ParsedSource {
  const source = rawSource.replace(/\r\n?/g, "\n");

  return (
    parseDelimitedFrontMatter(source, /^---\n([\s\S]*?)\n---\n?/, (value) => YAML.parse(value) ?? {}) ??
    parseDelimitedFrontMatter(source, /^\+\+\+\n([\s\S]*?)\n\+\+\+\n?/, (value) => parseToml(value) ?? {}) ??
    parseDelimitedFrontMatter(source, /^;;;\n([\s\S]*?)\n;;;\n?/, parseJsonFrontMatter) ?? {
      frontMatter: {},
      content: source,
    }
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item.trim() : ""))
      .filter(Boolean);
  }

  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }

  return [];
}

function extractTextPreview(content: string): string {
  return content
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\\\([\s\S]*?\\\)/g, " ")
    .replace(/\\\[[\s\S]*?\\\]/g, " ")
    .replace(/\$[^$\n]+\$/g, " ")
    .replace(/\\[A-Za-z]+/g, " ")
    .replace(/[{}]/g, " ")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]+`/g, " ")
    .replace(/!\[[^\]]*]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/\[\^[^\]]+\]/g, " ")
    .replace(/[#>*_~|-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function countWords(content: string): number {
  const matches = extractTextPreview(content).match(/[\p{L}\p{N}_-]+/gu);
  return matches?.length ?? 0;
}

function estimateReadingMinutes(wordCount: number): number {
  return Math.max(1, Math.round(wordCount / 225));
}

function stripInlineFormatting(value: string): string {
  return value
    .replace(/!\[[^\]]*]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/[*_~]+/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&[A-Za-z0-9#]+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractFirstHeading(content: string): string | null {
  const lines = content.replace(/\r\n?/g, "\n").split("\n");
  let activeFence: string | null = null;

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index];
    const trimmedLine = rawLine.trim();

    if (/^(```|~~~)/.test(trimmedLine)) {
      const fence = trimmedLine.slice(0, 3);
      activeFence = activeFence === fence ? null : fence;
      continue;
    }

    if (activeFence || !trimmedLine) {
      continue;
    }

    const atxMatch = trimmedLine.match(/^#{1,6}\s+(.+?)\s*#*\s*$/);

    if (atxMatch) {
      return stripInlineFormatting(atxMatch[1]) || null;
    }

    const underline = lines[index + 1]?.trim();

    if (underline && (/^=+$/.test(underline) || /^-+$/.test(underline))) {
      return stripInlineFormatting(trimmedLine) || null;
    }
  }

  return null;
}

function normalizeDate(value: unknown): string | null {
  if (value instanceof Date && !Number.isNaN(value.valueOf())) {
    return value.toISOString();
  }

  if (typeof value !== "string") {
    return null;
  }

  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? null : parsed.toISOString();
}

function extractDate(relativePath: string, frontMatter: Record<string, unknown>, stat: fs.Stats): string | null {
  const fromFrontMatter =
    normalizeDate(frontMatter.date) ??
    normalizeDate(frontMatter.published) ??
    normalizeDate(frontMatter.publishedAt) ??
    normalizeDate(frontMatter.updatedAt);

  if (fromFrontMatter) {
    return fromFrontMatter;
  }

  const fileName = path.basename(relativePath);
  const match = fileName.match(/^(\d{4}-\d{2}-\d{2})/);

  if (match) {
    const parsed = new Date(match[1]);

    if (!Number.isNaN(parsed.valueOf())) {
      return parsed.toISOString();
    }
  }

  return stat.mtime.toISOString();
}

function toTopicTag(label: string): TopicTag {
  return {
    slug: normalizeSlugSegment(label),
    label: humanizeToken(label),
  };
}

function dedupeTags(tags: TopicTag[]): TopicTag[] {
  const seen = new Set<string>();
  const unique: TopicTag[] = [];

  for (const tag of tags) {
    if (!seen.has(tag.slug)) {
      seen.add(tag.slug);
      unique.push(tag);
    }
  }

  return unique;
}

function deriveCollection(relativePath: string): TopicTag {
  const segments = toPosixPath(relativePath).split("/").filter(Boolean);
  const firstSegment = segments.length > 1 ? segments[0] : "root";
  return toTopicTag(firstSegment);
}

function getFolderRelativePath(relativePath: string): string {
  const directory = toPosixPath(path.dirname(relativePath));
  return directory === "." ? "" : directory;
}

function getFolderLabel(relativeDirectoryPath: string, collection: TopicTag): string {
  if (!relativeDirectoryPath) {
    return collection.label;
  }

  const segments = relativeDirectoryPath.split("/").filter(Boolean);
  return humanizeToken(segments.at(-1) ?? collection.label);
}

function deriveTags(relativePath: string, frontMatter: Record<string, unknown>): TopicTag[] {
  const explicitValues = [
    ...asStringArray(frontMatter.tags),
    ...asStringArray(frontMatter.tag),
    ...asStringArray(frontMatter.topic),
    ...asStringArray(frontMatter.topics),
    ...asStringArray(frontMatter.category),
    ...asStringArray(frontMatter.categories),
    ...asStringArray(frontMatter.series),
    ...asStringArray(frontMatter.subject),
    ...asStringArray(frontMatter.lang),
  ];

  const relativeSegments = toPosixPath(relativePath)
    .split("/")
    .filter(Boolean)
    .slice(0, -1)
    .filter((segment) => !STRUCTURAL_SEGMENTS.has(segment.toLowerCase()));
  const collection = deriveCollection(relativePath);
  const pathTags = relativeSegments.map(toTopicTag);
  const languageTag = relativeSegments.find((segment) => LANGUAGE_SEGMENTS.has(segment.toLowerCase()));
  const languageValues = languageTag ? [toTopicTag(languageTag)] : [];

  return dedupeTags([collection, ...pathTags, ...languageValues, ...explicitValues.map(toTopicTag)]);
}

function resolveAssetReference(relativeDocumentPath: string, rawReference: unknown): string | null {
  if (typeof rawReference !== "string" || !rawReference.trim()) {
    return null;
  }

  if (/^(https?:)?\/\//i.test(rawReference) || rawReference.startsWith("/")) {
    return rawReference;
  }

  const absolutePath = path.resolve(path.join(CONTENT_ROOT, path.dirname(relativeDocumentPath)), rawReference);

  if (!isWithinContentRoot(absolutePath)) {
    return null;
  }

  return relativeAssetPathToRoutePath(toPosixPath(path.relative(CONTENT_ROOT, absolutePath)));
}

function buildDocumentSummary(content: string, frontMatter: Record<string, unknown>): string {
  const preferredSummary =
    typeof frontMatter.description === "string"
      ? frontMatter.description
      : typeof frontMatter.summary === "string"
        ? frontMatter.summary
        : typeof frontMatter.excerpt === "string"
          ? frontMatter.excerpt
          : null;

  if (preferredSummary) {
    return preferredSummary.trim();
  }

  const preview = extractTextPreview(content);
  return preview.length > 220 ? `${preview.slice(0, 217).trimEnd()}...` : preview;
}

function buildFeatureFlags(content: string): FeatureFlags {
  return {
    hasMath: /\$\$|\\\(|\\\[|\\begin\{/.test(content),
    hasRawHtml: /<[A-Za-z][^>]*>/.test(content),
    hasMermaid: /```mermaid/i.test(content),
  };
}

function sortDocuments(documents: DocumentMeta[]): DocumentMeta[] {
  return [...documents].sort((left, right) => {
    const leftTime = left.publishedAt ? new Date(left.publishedAt).valueOf() : 0;
    const rightTime = right.publishedAt ? new Date(right.publishedAt).valueOf() : 0;

    if (leftTime !== rightTime) {
      return rightTime - leftTime;
    }

    return left.title.localeCompare(right.title);
  });
}

function toReference(document: DocumentMeta): DocumentReference {
  return {
    title: document.title,
    url: document.url,
    collectionLabel: document.collection.label,
  };
}

function pushGroupedValue<T>(groups: Map<string, T[]>, key: string, value: T): void {
  const current = groups.get(key) ?? [];
  current.push(value);
  groups.set(key, current);
}

function getDocumentMeta(file: MarkdownFileDescriptor): DocumentMeta | null {
  const cached = documentRecordCache.get(file.relativePath);

  if (cached?.versionKey === file.versionKey) {
    return cached.document;
  }

  try {
    const rawSource = fs.readFileSync(file.filePath, "utf8");
    const { frontMatter, content } = parseSource(rawSource);
    const stat = fs.statSync(file.filePath);
    const wordCount = countWords(content);
    const collection = deriveCollection(file.relativePath);
    const folderRelativePath = getFolderRelativePath(file.relativePath);
    const title =
      (typeof frontMatter.title === "string" && frontMatter.title.trim()) ||
      extractFirstHeading(content) ||
      humanizeToken(path.basename(file.relativePath, path.extname(file.relativePath)));

    const document = {
      filePath: file.filePath,
      relativePath: file.relativePath,
      routeSegments: relativeFilePathToDocRouteSegments(file.relativePath),
      url: relativeFilePathToDocRoutePath(file.relativePath),
      collection,
      folderRelativePath,
      folderLabel: getFolderLabel(folderRelativePath, collection),
      folderUrl: relativeDirectoryPathToFolderRoutePath(folderRelativePath),
      title,
      summary: buildDocumentSummary(content, frontMatter),
      authors: [...asStringArray(frontMatter.author), ...asStringArray(frontMatter.authors)],
      publishedAt: extractDate(file.relativePath, frontMatter, stat),
      heroImage:
        resolveAssetReference(file.relativePath, frontMatter.image) ??
        resolveAssetReference(file.relativePath, frontMatter.heroImage) ??
        resolveAssetReference(file.relativePath, frontMatter.cover),
      tags: deriveTags(file.relativePath, frontMatter),
      wordCount,
      readingMinutes: estimateReadingMinutes(wordCount),
      features: buildFeatureFlags(content),
    } satisfies DocumentMeta;

    documentRecordCache.set(file.relativePath, {
      versionKey: file.versionKey,
      document,
    });

    return document;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`[content] Skipping ${file.relativePath}: ${message}`);
    return null;
  }
}

function buildCollectionSummaries(documentsByCollection: Map<string, DocumentMeta[]>): CollectionSummary[] {
  return [...documentsByCollection.entries()]
    .map(([slug, documents]) => {
      const tagCounts = new Map<string, { label: string; count: number }>();

      for (const document of documents) {
        for (const tag of document.tags) {
          if (tag.slug === slug) {
            continue;
          }

          const current = tagCounts.get(tag.slug) ?? { label: tag.label, count: 0 };
          current.count += 1;
          tagCounts.set(tag.slug, current);
        }
      }

      const topTags = [...tagCounts.entries()]
        .sort((left, right) => right[1].count - left[1].count || left[1].label.localeCompare(right[1].label))
        .slice(0, 4)
        .map(([tagSlug, value]) => ({
          slug: tagSlug,
          label: value.label,
        }));

      return {
        slug,
        label: documents[0]?.collection.label ?? humanizeToken(slug),
        count: documents.length,
        latestDocument: documents[0] ? toReference(documents[0]) : null,
        topTags,
      } satisfies CollectionSummary;
    })
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
}

function toFolderReference(relativePath: string): FolderReference {
  return {
    label: humanizeToken(relativePath.split("/").at(-1) ?? relativePath),
    url: relativeDirectoryPathToFolderRoutePath(relativePath),
    relativePath,
  };
}

function buildFolderSummaries(documentsByFolder: Map<string, DocumentMeta[]>): FolderSummary[] {
  const folderPaths = [...documentsByFolder.keys()].filter(Boolean);
  const childFolderMap = new Map<string, string[]>();

  for (const relativePath of folderPaths) {
    const parentPath = relativePath.split("/").slice(0, -1).join("/");

    if (parentPath) {
      pushGroupedValue(childFolderMap, parentPath, relativePath);
    }
  }

  return folderPaths
    .map((relativePath) => {
      const documents = documentsByFolder.get(relativePath) ?? [];
      const collection = documents[0]?.collection ?? toTopicTag(relativePath.split("/")[0] ?? "root");
      const childFolders = (childFolderMap.get(relativePath) ?? [])
        .sort((left, right) => left.localeCompare(right))
        .slice(0, 6)
        .map(toFolderReference);

      return {
        relativePath,
        url: relativeDirectoryPathToFolderRoutePath(relativePath),
        label: getFolderLabel(relativePath, collection),
        pathLabel: relativePath,
        collection,
        depth: relativePath.split("/").filter(Boolean).length,
        documentCount: documents.length,
        latestDocument: documents[0] ? toReference(documents[0]) : null,
        sampleDocuments: documents.slice(0, 3).map(toReference),
        childFolders,
      } satisfies FolderSummary;
    })
    .sort(
      (left, right) =>
        left.depth - right.depth ||
        right.documentCount - left.documentCount ||
        left.pathLabel.localeCompare(right.pathLabel),
    );
}

function buildTagSummaries(documentsByTag: Map<string, DocumentMeta[]>): TagSummary[] {
  return [...documentsByTag.entries()]
    .map(([slug, documents]) => ({
      slug,
      label: documents[0]?.tags.find((tag) => tag.slug === slug)?.label ?? humanizeToken(slug),
      count: documents.length,
      sampleDocuments: documents.slice(0, 3).map(toReference),
    }))
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
}

function pruneStaleCaches(files: MarkdownFileDescriptor[], state: RepositoryState): void {
  const validRelativePaths = new Set(files.map((file) => file.relativePath));
  const validRouteKeys = new Set(state.documents.map((document) => document.routeSegments.join("/")));

  for (const relativePath of [...documentRecordCache.keys()]) {
    if (!validRelativePaths.has(relativePath)) {
      documentRecordCache.delete(relativePath);
    }
  }

  for (const routeKey of [...documentPageCache.keys()]) {
    if (!validRouteKeys.has(routeKey)) {
      documentPageCache.delete(routeKey);
    }
  }
}

function buildRepositoryState(): RepositoryState {
  const files = listMarkdownFiles();
  const snapshotKey = buildSnapshotKey(files);

  if (repositoryStateCache?.snapshotKey === snapshotKey) {
    return repositoryStateCache;
  }

  const documents = sortDocuments(files.map(getDocumentMeta).filter((document): document is DocumentMeta => Boolean(document)));
  const documentByRouteKey = new Map<string, DocumentMeta>();
  const fileVersionByRelativePath = new Map<string, string>();
  const documentsByCollection = new Map<string, DocumentMeta[]>();
  const documentsByFolder = new Map<string, DocumentMeta[]>();
  const documentsByTag = new Map<string, DocumentMeta[]>();
  const fileDescriptorByRelativePath = new Map(files.map((file) => [file.relativePath, file]));

  for (const document of documents) {
    const routeKey = document.routeSegments.join("/");
    documentByRouteKey.set(routeKey, document);
    fileVersionByRelativePath.set(
      document.relativePath,
      fileDescriptorByRelativePath.get(document.relativePath)?.versionKey ?? document.relativePath,
    );
    pushGroupedValue(documentsByCollection, document.collection.slug, document);
    pushGroupedValue(documentsByFolder, document.folderRelativePath, document);

    for (const tag of document.tags) {
      pushGroupedValue(documentsByTag, tag.slug, document);
    }
  }

  const collections = buildCollectionSummaries(documentsByCollection);
  const folders = buildFolderSummaries(documentsByFolder);
  const tags = buildTagSummaries(documentsByTag);
  const state = {
    snapshotKey,
    documents,
    documentByRouteKey,
    fileVersionByRelativePath,
    documentsByCollection,
    documentsByFolder,
    documentsByTag,
    collections,
    collectionBySlug: new Map(collections.map((collection) => [collection.slug, collection])),
    folders,
    folderByPath: new Map(folders.map((folder) => [folder.relativePath, folder])),
    tags,
    tagBySlug: new Map(tags.map((tag) => [tag.slug, tag])),
  } satisfies RepositoryState;

  pruneStaleCaches(files, state);
  repositoryStateCache = state;
  return state;
}

function getRepositoryState(): RepositoryState {
  if (!DEVELOPMENT_MODE) {
    return repositoryStateCache ?? buildRepositoryState();
  }

  const now = Date.now();

  if (repositoryStateCache && now - lastRepositoryStateCheckAt < DEVELOPMENT_CACHE_TTL_MS) {
    return repositoryStateCache;
  }

  lastRepositoryStateCheckAt = now;
  return buildRepositoryState();
}

async function loadDocumentPage(routeSegments: string[]): Promise<DocumentPageData | null> {
  const state = getRepositoryState();
  const routeKey = routeSegments.join("/");
  const document = state.documentByRouteKey.get(routeKey);

  if (!document) {
    return null;
  }

  const versionKey = state.fileVersionByRelativePath.get(document.relativePath) ?? document.relativePath;
  const cached = documentPageCache.get(routeKey);

  if (cached?.versionKey === versionKey) {
    return cached.page;
  }

  const page = fs.promises
    .readFile(document.filePath, "utf8")
    .then(async (rawSource) => {
      const { frontMatter, content } = parseSource(rawSource);
      const rendered = await renderMarkdownDocument(content, document.relativePath);

      return {
        ...document,
        frontMatter,
        content,
        html: rendered.html,
        headings: rendered.headings,
      } satisfies DocumentPageData;
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[content] Failed to render ${document.relativePath}: ${message}`);
      return null;
    });

  documentPageCache.set(routeKey, {
    versionKey,
    page,
  });

  return page;
}

export function getAllDocuments(): DocumentMeta[] {
  return getRepositoryState().documents;
}

export function getFolderSummaries(): FolderSummary[] {
  return getRepositoryState().folders;
}

export function getCollectionSummaries(): CollectionSummary[] {
  return getRepositoryState().collections;
}

export function getCollectionBySlug(slug: string): CollectionSummary | null {
  return getRepositoryState().collectionBySlug.get(slug) ?? null;
}

export function getDocumentsByCollection(slug: string): DocumentMeta[] {
  return getRepositoryState().documentsByCollection.get(slug) ?? [];
}

export function getFolderByPath(folderSegments: string[]): FolderSummary | null {
  const relativePath = toPosixPath(folderSegments.join("/"));
  return getRepositoryState().folderByPath.get(relativePath) ?? null;
}

export function getDocumentsByFolderPath(relativePath: string): DocumentMeta[] {
  return getRepositoryState().documentsByFolder.get(relativePath) ?? [];
}

export function getTagSummaries(): TagSummary[] {
  return getRepositoryState().tags;
}

export function getTagBySlug(slug: string): TagSummary | null {
  return getRepositoryState().tagBySlug.get(slug) ?? null;
}

export function getDocumentsByTag(slug: string): DocumentMeta[] {
  return getRepositoryState().documentsByTag.get(slug) ?? [];
}

export function getRelatedDocuments(document: DocumentMeta, limit = 3): DocumentMeta[] {
  const currentTags = new Set(document.tags.map((tag) => tag.slug));

  return getRepositoryState()
    .documents.filter((candidate) => candidate.url !== document.url)
    .map((candidate) => {
      const sharedTags = candidate.tags.filter((tag) => currentTags.has(tag.slug)).length;
      const collectionBonus = candidate.collection.slug === document.collection.slug ? 3 : 0;

      return {
        candidate,
        score: sharedTags + collectionBonus,
      };
    })
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score || left.candidate.title.localeCompare(right.candidate.title))
    .slice(0, limit)
    .map((entry) => entry.candidate);
}

export function getMoreDocumentsFromSameFolder(document: DocumentMeta, limit = 4): DocumentMeta[] {
  return getDocumentsByFolderPath(document.folderRelativePath)
    .filter((candidate) => candidate.url !== document.url)
    .slice(0, limit);
}

export async function getDocumentBySlug(routeSegments: string[]): Promise<DocumentPageData | null> {
  return loadDocumentPage(routeSegments);
}
