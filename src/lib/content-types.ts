export interface TopicTag {
  slug: string;
  label: string;
}

export interface FeatureFlags {
  hasMath: boolean;
  hasRawHtml: boolean;
  hasMermaid: boolean;
}

export interface DocumentHeading {
  id: string;
  text: string;
  depth: number;
}

export interface DocumentReference {
  title: string;
  url: string;
  collectionLabel: string;
}

export interface FolderReference {
  label: string;
  url: string;
  relativePath: string;
}

export interface DocumentMeta {
  filePath: string;
  relativePath: string;
  routeSegments: string[];
  url: string;
  collection: TopicTag;
  folderRelativePath: string;
  folderLabel: string;
  folderUrl: string;
  title: string;
  summary: string;
  authors: string[];
  publishedAt: string | null;
  heroImage: string | null;
  tags: TopicTag[];
  wordCount: number;
  readingMinutes: number;
  features: FeatureFlags;
}

export interface DocumentPageData extends DocumentMeta {
  frontMatter: Record<string, unknown>;
  content: string;
  html: string;
  headings: DocumentHeading[];
}

export interface TagSummary {
  slug: string;
  label: string;
  count: number;
  sampleDocuments: DocumentReference[];
}

export interface CollectionSummary {
  slug: string;
  label: string;
  count: number;
  latestDocument: DocumentReference | null;
  topTags: TopicTag[];
}

export interface FolderSummary {
  relativePath: string;
  url: string;
  label: string;
  pathLabel: string;
  collection: TopicTag;
  depth: number;
  documentCount: number;
  latestDocument: DocumentReference | null;
  sampleDocuments: DocumentReference[];
  childFolders: FolderReference[];
}
