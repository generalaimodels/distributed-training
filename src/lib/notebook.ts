import { parse as parseToml } from "smol-toml";
import YAML from "yaml";

import type { DocumentHeading, FeatureFlags, NotebookMetrics, NotebookSection } from "@/lib/content-types";
import { renderMarkdownDocument } from "@/lib/markdown";

interface NotebookFile {
  cells: NotebookCell[];
  metadata?: Record<string, unknown>;
  nbformat?: number;
  nbformat_minor?: number;
}

interface NotebookCell {
  cell_type?: unknown;
  execution_count?: unknown;
  metadata?: Record<string, unknown>;
  outputs?: NotebookOutput[];
  source?: string | string[];
}

interface NotebookOutput {
  data?: Record<string, unknown>;
  ename?: unknown;
  evalue?: unknown;
  metadata?: Record<string, unknown>;
  name?: unknown;
  output_type?: unknown;
  text?: unknown;
  traceback?: unknown;
}

interface ParsedSource {
  frontMatter: Record<string, unknown>;
  content: string;
}

interface NormalizedNotebookCell {
  cellType: "markdown" | "code" | "raw";
  executionCount: number | null;
  metadata: Record<string, unknown>;
  outputs: NotebookOutput[];
  source: string;
}

export interface NotebookSummary {
  authors: string[];
  features: FeatureFlags;
  frontMatter: Record<string, unknown>;
  heroImageReference: string | null;
  metrics: NotebookMetrics;
  summary: string;
  title: string | null;
  wordCount: number;
}

export interface RenderedNotebookDocument {
  content: string;
  frontMatter: Record<string, unknown>;
  headings: DocumentHeading[];
  html: string;
  notebookHeadingAliasMap: Record<string, string>;
  notebookHeadingSectionMap: Record<string, number>;
  notebookSections: NotebookSection[];
}

const IMAGE_MIME_PREFIXES = new Map<string, string>([
  ["image/png", "png"],
  ["image/jpeg", "jpeg"],
  ["image/gif", "gif"],
  ["image/webp", "webp"],
]);
const MIME_RENDER_PRIORITY = [
  "text/html",
  "text/markdown",
  "image/svg+xml",
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
  "application/json",
  "text/plain",
  "text/latex",
];
const NOTEBOOK_SECTION_PREFIX = "notebook-cell";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeTextValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => (typeof item === "string" ? item : "")).join("").replace(/\r\n?/g, "\n");
  }

  if (typeof value === "string") {
    return value.replace(/\r\n?/g, "\n");
  }

  return "";
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

function extractFirstHeading(content: string): string | null {
  const lines = content.replace(/\r\n?/g, "\n").split("\n");
  let activeFence: string | null = null;

  for (let index = 0; index < lines.length; index += 1) {
    const trimmedLine = lines[index].trim();

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

function truncateSummary(value: string, maxLength = 220): string {
  const normalized = value.trim();
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 3).trimEnd()}...` : normalized;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttribute(value: string): string {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function sanitizeRichHtml(value: string): string {
  return value.replace(/<script\b[\s\S]*?<\/script>/gi, "");
}

function normalizeNotebook(rawSource: string): { cells: NormalizedNotebookCell[]; metadata: Record<string, unknown> } {
  const parsed = JSON.parse(rawSource) as unknown;

  if (!isPlainObject(parsed)) {
    throw new Error("Notebook file must be a JSON object.");
  }

  const notebook = parsed as unknown as NotebookFile;
  const cells = Array.isArray(notebook.cells) ? notebook.cells : [];

  return {
    metadata: isPlainObject(notebook.metadata) ? notebook.metadata : {},
    cells: cells.map((cell) => {
      const metadata = isPlainObject(cell.metadata) ? cell.metadata : {};
      const rawCellType = typeof cell.cell_type === "string" ? cell.cell_type.toLowerCase() : "raw";
      const cellType = rawCellType === "markdown" || rawCellType === "code" ? rawCellType : "raw";
      const executionCount = typeof cell.execution_count === "number" ? cell.execution_count : null;

      return {
        cellType,
        executionCount,
        metadata,
        outputs: Array.isArray(cell.outputs) ? cell.outputs : [],
        source: normalizeTextValue(cell.source),
      } satisfies NormalizedNotebookCell;
    }),
  };
}

function extractNotebookFrontMatter(
  cells: NormalizedNotebookCell[],
): { cleanedCells: NormalizedNotebookCell[]; frontMatter: Record<string, unknown> } {
  const markdownIndex = cells.findIndex((cell) => cell.cellType === "markdown" && cell.source.trim().length > 0);

  if (markdownIndex < 0) {
    return {
      cleanedCells: cells,
      frontMatter: {},
    };
  }

  const targetCell = cells[markdownIndex];
  const parsed = parseSource(targetCell.source);

  if (Object.keys(parsed.frontMatter).length === 0) {
    return {
      cleanedCells: cells,
      frontMatter: {},
    };
  }

  const cleanedCells = [...cells];
  cleanedCells[markdownIndex] = {
    ...targetCell,
    source: parsed.content.trimStart(),
  };

  return {
    cleanedCells,
    frontMatter: parsed.frontMatter,
  };
}

function extractNotebookTitle(
  frontMatter: Record<string, unknown>,
  metadata: Record<string, unknown>,
  cells: NormalizedNotebookCell[],
): string | null {
  const explicitTitle =
    (typeof frontMatter.title === "string" && frontMatter.title.trim()) ||
    (typeof metadata.title === "string" && metadata.title.trim()) ||
    (typeof metadata.name === "string" && metadata.name.trim()) ||
    (typeof metadata.colab === "object" && isPlainObject(metadata.colab) && typeof metadata.colab.name === "string"
      ? metadata.colab.name.trim()
      : "");

  if (explicitTitle) {
    return explicitTitle;
  }

  for (const cell of cells) {
    if (cell.cellType !== "markdown") {
      continue;
    }

    const heading = extractFirstHeading(cell.source);

    if (heading) {
      return heading;
    }
  }

  return null;
}

function extractNotebookHeroImage(frontMatter: Record<string, unknown>, cells: NormalizedNotebookCell[]): string | null {
  for (const key of ["image", "heroImage", "cover"]) {
    if (typeof frontMatter[key] === "string" && frontMatter[key].trim()) {
      return frontMatter[key].trim();
    }
  }

  for (const cell of cells) {
    if (cell.cellType !== "markdown") {
      continue;
    }

    const markdownImageMatch = cell.source.match(/!\[[^\]]*]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/);

    if (markdownImageMatch?.[1]) {
      return markdownImageMatch[1];
    }

    const htmlImageMatch = cell.source.match(/<img\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/i);

    if (htmlImageMatch?.[1]) {
      return htmlImageMatch[1];
    }
  }

  return null;
}

function buildNotebookMetrics(cells: NormalizedNotebookCell[], metadata: Record<string, unknown>): NotebookMetrics {
  const kernelspec = isPlainObject(metadata.kernelspec) ? metadata.kernelspec : {};
  const languageInfo = isPlainObject(metadata.language_info) ? metadata.language_info : {};

  return {
    cellCount: cells.length,
    codeCellCount: cells.filter((cell) => cell.cellType === "code").length,
    markdownCellCount: cells.filter((cell) => cell.cellType === "markdown").length,
    rawCellCount: cells.filter((cell) => cell.cellType === "raw").length,
    hasOutputs: cells.some((cell) => cell.outputs.length > 0),
    language:
      (typeof languageInfo.name === "string" && languageInfo.name.trim()) ||
      (typeof kernelspec.language === "string" && kernelspec.language.trim()) ||
      null,
    kernelDisplayName: typeof kernelspec.display_name === "string" ? kernelspec.display_name.trim() || null : null,
    isColab: isPlainObject(metadata.colab),
  };
}

function buildNotebookFeatureFlags(cells: NormalizedNotebookCell[]): FeatureFlags {
  const combinedText = cells.map((cell) => cell.source).join("\n\n");
  const outputText = cells
    .flatMap((cell) => cell.outputs)
    .map((output) => {
      if (isPlainObject(output) && isPlainObject(output.data)) {
        return Object.values(output.data).map(normalizeTextValue).join("\n");
      }

      if (isPlainObject(output) && "text" in output) {
        return normalizeTextValue(output.text);
      }

      if (isPlainObject(output) && "traceback" in output) {
        return normalizeTextValue(output.traceback);
      }

      return "";
    })
    .join("\n");
  const fullText = `${combinedText}\n${outputText}`;

  return {
    hasMath: /\$\$|\\\(|\\\[|\\begin\{/.test(fullText),
    hasRawHtml: /<[A-Za-z][^>]*>/.test(fullText) || /text\/html|image\/svg\+xml/.test(outputText),
    hasMermaid: /```mermaid/i.test(combinedText),
  };
}

function buildNotebookSummaryText(frontMatter: Record<string, unknown>, cells: NormalizedNotebookCell[]): string {
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

  const markdownText = cells
    .filter((cell) => cell.cellType === "markdown")
    .map((cell) => cell.source)
    .join("\n\n");
  const preview = extractTextPreview(markdownText);

  if (preview) {
    return truncateSummary(preview);
  }

  const codePreview = cells
    .filter((cell) => cell.cellType === "code")
    .map((cell) => cell.source)
    .join("\n\n")
    .trim();

  if (codePreview) {
    return truncateSummary(codePreview.replace(/\s+/g, " "));
  }

  return "Interactive notebook content with code cells, markdown explanations, and rich outputs.";
}

function buildNotebookWordCount(cells: NormalizedNotebookCell[]): number {
  const readableText = cells
    .filter((cell) => cell.cellType !== "code")
    .map((cell) => cell.source)
    .join("\n\n");

  return countWords(readableText);
}

function getNotebookCellLanguage(cell: NormalizedNotebookCell, fallbackLanguage: string | null): string {
  const metadata = isPlainObject(cell.metadata) ? cell.metadata : {};

  if (typeof metadata.language === "string" && metadata.language.trim()) {
    return metadata.language.trim();
  }

  if (typeof metadata.magics_language === "string" && metadata.magics_language.trim()) {
    return metadata.magics_language.trim();
  }

  const vscode = isPlainObject(metadata.vscode) ? metadata.vscode : {};

  if (typeof vscode.languageId === "string" && vscode.languageId.trim()) {
    return vscode.languageId.trim();
  }

  return fallbackLanguage ?? "text";
}

function renderNotebookOutputHeader(label: string): string {
  return `<div class="notebook-output-header"><span class="notebook-output-chip">${escapeHtml(label)}</span></div>`;
}

function normalizeMimeBundleValue(value: unknown): string {
  return normalizeTextValue(value);
}

async function renderMimeBundle(
  bundle: Record<string, unknown>,
  relativeDocumentPath: string,
): Promise<{ html: string; label: string } | null> {
  for (const mimeType of MIME_RENDER_PRIORITY) {
    if (!(mimeType in bundle)) {
      continue;
    }

    const value = bundle[mimeType];

    if (mimeType === "text/html") {
      const html = normalizeMimeBundleValue(value);

      if (!html.trim()) {
        continue;
      }

      return {
        html: `<div class="notebook-output-rich">${sanitizeRichHtml(html)}</div>`,
        label: "HTML output",
      };
    }

    if (mimeType === "text/markdown") {
      const markdown = normalizeMimeBundleValue(value);

      if (!markdown.trim()) {
        continue;
      }

      const rendered = await renderMarkdownDocument(markdown, relativeDocumentPath);

      return {
        html: `<div class="notebook-output-markdown">${rendered.html}</div>`,
        label: "Markdown output",
      };
    }

    if (mimeType === "image/svg+xml") {
      const svg = normalizeMimeBundleValue(value);

      if (!svg.trim()) {
        continue;
      }

      return {
        html: `<div class="notebook-output-image notebook-output-image-vector">${sanitizeRichHtml(svg)}</div>`,
        label: "SVG output",
      };
    }

    if (IMAGE_MIME_PREFIXES.has(mimeType)) {
      const encoded = normalizeMimeBundleValue(value).replace(/\s+/g, "");

      if (!encoded) {
        continue;
      }

      return {
        html: `<div class="notebook-output-image"><img src="data:${mimeType};base64,${encoded}" alt="${escapeAttribute(mimeType)}" loading="lazy" /></div>`,
        label: "Image output",
      };
    }

    if (mimeType === "application/json") {
      const jsonValue = typeof value === "string" ? value : JSON.stringify(value, null, 2);

      if (!jsonValue?.trim()) {
        continue;
      }

      return {
        html: `<pre class="notebook-output-pre"><code>${escapeHtml(jsonValue)}</code></pre>`,
        label: "JSON output",
      };
    }

    if (mimeType === "text/latex" || mimeType === "text/plain") {
      const text = normalizeMimeBundleValue(value);

      if (!text.trim()) {
        continue;
      }

      return {
        html: `<pre class="notebook-output-pre"><code>${escapeHtml(text)}</code></pre>`,
        label: mimeType === "text/latex" ? "LaTeX output" : "Text output",
      };
    }
  }

  return null;
}

async function renderNotebookOutputs(outputs: NotebookOutput[], relativeDocumentPath: string): Promise<string> {
  const renderedGroups = await Promise.all(
    outputs.map(async (output) => {
      if (!isPlainObject(output)) {
        return "";
      }

      const outputType = typeof output.output_type === "string" ? output.output_type : "";

      if (outputType === "stream") {
        const label = typeof output.name === "string" && output.name.trim() ? `${output.name} stream` : "Stream output";
        const text = normalizeTextValue(output.text);

        if (!text.trim()) {
          return "";
        }

        return `<div class="notebook-output-group">${renderNotebookOutputHeader(label)}<pre class="notebook-output-pre"><code>${escapeHtml(text)}</code></pre></div>`;
      }

      if (outputType === "error") {
        const traceback = normalizeTextValue(output.traceback);
        const errorName = typeof output.ename === "string" ? output.ename : "Error";
        const errorValue = typeof output.evalue === "string" ? `: ${output.evalue}` : "";
        const content = traceback.trim() || `${errorName}${errorValue}`;

        return `<div class="notebook-output-group notebook-output-group-error">${renderNotebookOutputHeader(`Error: ${errorName}`)}<pre class="notebook-output-pre"><code>${escapeHtml(content)}</code></pre></div>`;
      }

      if ((outputType === "display_data" || outputType === "execute_result") && isPlainObject(output.data)) {
        const renderedBundle = await renderMimeBundle(output.data, relativeDocumentPath);

        if (!renderedBundle) {
          return "";
        }

        return `<div class="notebook-output-group">${renderNotebookOutputHeader(renderedBundle.label)}${renderedBundle.html}</div>`;
      }

      return "";
    }),
  );

  const html = renderedGroups.filter(Boolean).join("");
  return html ? `<div class="notebook-output-stack">${html}</div>` : "";
}

function wrapNotebookSection(index: number, kind: NotebookSection["kind"], innerHtml: string): string {
  return `<section id="${NOTEBOOK_SECTION_PREFIX}-${index + 1}" class="notebook-cell notebook-cell-${kind}" data-notebook-section="${index + 1}" data-notebook-kind="${kind}">${innerHtml}</section>`;
}

function rewriteFragmentLinks(html: string, headingAliasMap: Record<string, string>): string {
  return html.replace(/href="#([^"]+)"/g, (match, rawValue: string) => {
    const decodedValue = decodeURIComponent(rawValue);
    const mappedId = headingAliasMap[decodedValue] ?? headingAliasMap[rawValue];

    if (!mappedId) {
      return match;
    }

    return `href="#${escapeAttribute(mappedId)}"`;
  });
}

export function summarizeNotebookDocument(rawSource: string): NotebookSummary {
  const notebook = normalizeNotebook(rawSource);
  const { cleanedCells, frontMatter } = extractNotebookFrontMatter(notebook.cells);
  const metrics = buildNotebookMetrics(cleanedCells, notebook.metadata);
  const wordCount = buildNotebookWordCount(cleanedCells);
  const estimatedWordCount =
    wordCount > 0 ? wordCount : Math.max(180, metrics.markdownCellCount * 90 + metrics.codeCellCount * 36 + metrics.rawCellCount * 48);

  return {
    authors: [
      ...asStringArray(frontMatter.author),
      ...asStringArray(frontMatter.authors),
      ...asStringArray(notebook.metadata.author),
      ...asStringArray(notebook.metadata.authors),
    ],
    features: buildNotebookFeatureFlags(cleanedCells),
    frontMatter,
    heroImageReference: extractNotebookHeroImage(frontMatter, cleanedCells),
    metrics,
    summary: buildNotebookSummaryText(frontMatter, cleanedCells),
    title: extractNotebookTitle(frontMatter, notebook.metadata, cleanedCells),
    wordCount: estimatedWordCount,
  };
}

export async function renderNotebookDocument(
  rawSource: string,
  relativeDocumentPath: string,
): Promise<RenderedNotebookDocument> {
  const notebook = normalizeNotebook(rawSource);
  const { cleanedCells, frontMatter } = extractNotebookFrontMatter(notebook.cells);
  const metrics = buildNotebookMetrics(cleanedCells, notebook.metadata);
  const notebookSections: NotebookSection[] = [];
  const headings: DocumentHeading[] = [];
  const notebookHeadingAliasMap: Record<string, string> = {};
  const notebookHeadingSectionMap: Record<string, number> = {};

  const renderedSections = await Promise.all(
    cleanedCells.map(async (cell, index) => {
      const sectionId = `${NOTEBOOK_SECTION_PREFIX}-${index + 1}`;
      const headingPrefix = `${sectionId}-heading`;

      if (cell.cellType === "markdown") {
        if (!cell.source.trim()) {
          return null;
        }

        const rendered = await renderMarkdownDocument(cell.source, relativeDocumentPath, {
          headingIdPrefix: headingPrefix,
        });

        return {
          headingPrefix,
          headings: rendered.headings,
          html: wrapNotebookSection(index, "markdown", rendered.html),
          kind: "markdown" as const,
          sectionId,
        };
      }

      if (cell.cellType === "code") {
        const language = getNotebookCellLanguage(cell, metrics.language);
        const promptLabel = cell.executionCount !== null ? `In [${cell.executionCount}]` : "Code";
        const sourceHtml = cell.source.trim()
          ? `<div class="notebook-code-shell"><div class="notebook-code-header"><span class="notebook-code-prompt">${escapeHtml(promptLabel)}</span><span class="notebook-code-language">${escapeHtml(language)}</span></div><pre class="notebook-code-pre"><code>${escapeHtml(cell.source)}</code></pre></div>`
          : "";
        const outputsHtml = await renderNotebookOutputs(cell.outputs, relativeDocumentPath);

        if (!sourceHtml && !outputsHtml) {
          return null;
        }

        return {
          headingPrefix,
          headings: [] as DocumentHeading[],
          html: wrapNotebookSection(index, "code", `${sourceHtml}${outputsHtml}`),
          kind: "code" as const,
          sectionId,
        };
      }

      if (!cell.source.trim()) {
        return null;
      }

      const rawHeader = `<div class="notebook-raw-header"><span class="notebook-raw-chip">Raw cell</span></div>`;

      return {
        headingPrefix,
        headings: [] as DocumentHeading[],
        html: wrapNotebookSection(
          index,
          "raw",
          `${rawHeader}<pre class="notebook-raw-pre"><code>${escapeHtml(cell.source)}</code></pre>`,
        ),
        kind: "raw" as const,
        sectionId,
      };
    }),
  );

  for (const renderedSection of renderedSections) {
    if (!renderedSection) {
      continue;
    }

    const sectionIndex = notebookSections.length;
    const sectionHeadingIds: string[] = [];

    for (const heading of renderedSection.headings) {
      headings.push(heading);
      sectionHeadingIds.push(heading.id);
      notebookHeadingSectionMap[heading.id] = sectionIndex;

      const rawId = heading.id.startsWith(`${renderedSection.headingPrefix}-`)
        ? heading.id.slice(renderedSection.headingPrefix.length + 1)
        : heading.id;

      notebookHeadingAliasMap[heading.id] = heading.id;

      if (!(rawId in notebookHeadingAliasMap)) {
        notebookHeadingAliasMap[rawId] = heading.id;
      }

      if (!(rawId in notebookHeadingSectionMap)) {
        notebookHeadingSectionMap[rawId] = sectionIndex;
      }
    }

    notebookSections.push({
      id: renderedSection.sectionId,
      html: renderedSection.html,
      headingIds: sectionHeadingIds,
      kind: renderedSection.kind,
    });
  }

  const rewrittenSections = notebookSections.map((section) => ({
    ...section,
    html: rewriteFragmentLinks(section.html, notebookHeadingAliasMap),
  }));

  return {
    content: cleanedCells.map((cell) => cell.source).join("\n\n"),
    frontMatter,
    headings,
    html: rewrittenSections.map((section) => section.html).join("\n"),
    notebookHeadingAliasMap,
    notebookHeadingSectionMap,
    notebookSections: rewrittenSections,
  };
}
