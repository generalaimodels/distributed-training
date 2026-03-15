import fs from "node:fs";
import path from "node:path";

import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeKatex from "rehype-katex";
import rehypePrettyCode from "rehype-pretty-code";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import type { Root as HastRoot } from "hast";
import type { Root as MdastRoot } from "mdast";
import { unified } from "unified";
import { visit } from "unist-util-visit";
import remarkDirective from "remark-directive";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";

import type { DocumentHeading } from "@/lib/content-types";
import {
  CONTENT_ROOT,
  isWithinContentRoot,
  relativeAssetPathToRoutePath,
  relativeDirectoryPathToFolderRoutePath,
  relativeFilePathToDocRoutePath,
  toPosixPath,
} from "@/lib/pathing";

const CALLOUT_TITLES: Record<string, string> = {
  note: "Note",
  info: "Info",
  warning: "Warning",
  tip: "Tip",
  important: "Important",
  caution: "Caution",
  success: "Success",
  error: "Error",
};
const CALLOUT_TAGS = new Set([
  ...Object.keys(CALLOUT_TITLES),
  "remark",
  "theorem",
  "lemma",
  "proof",
  "example",
]);
const SEMANTIC_DIRECTIVE_TAGS = new Set([
  "div",
  "span",
  "section",
  "article",
  "header",
  "footer",
  "main",
  "nav",
  "aside",
  "details",
  "summary",
  "figure",
  "figcaption",
  "blockquote",
  "ul",
  "ol",
  "li",
]);
const BLOCK_LAYOUT_TAGS = new Set([
  "address",
  "article",
  "aside",
  "blockquote",
  "details",
  "div",
  "dl",
  "fieldset",
  "figcaption",
  "figure",
  "footer",
  "form",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "header",
  "hr",
  "li",
  "main",
  "nav",
  "ol",
  "p",
  "pre",
  "section",
  "summary",
  "table",
  "tbody",
  "td",
  "tfoot",
  "th",
  "thead",
  "tr",
  "ul",
]);

function extractHastText(node: any): string {
  if (!node) {
    return "";
  }

  if (node.type === "text") {
    return node.value ?? "";
  }

  if (!Array.isArray(node.children)) {
    return "";
  }

  return node.children.map((child: any) => extractHastText(child)).join("");
}

function getClassNames(properties: Record<string, unknown> | undefined): string[] {
  const value = properties?.className;

  if (Array.isArray(value)) {
    return value.flatMap((item) =>
      String(item)
        .split(/\s+/)
        .map((className) => className.trim())
        .filter(Boolean),
    );
  }

  if (typeof value === "string") {
    return value
      .split(/\s+/)
      .map((className) => className.trim())
      .filter(Boolean);
  }

  return [];
}

function isWhitespaceTextNode(node: any): boolean {
  return node?.type === "text" && !/\S/.test(String(node.value ?? ""));
}

function trimNodeRun(nodes: any[]): any[] {
  let start = 0;
  let end = nodes.length;

  while (start < end && isWhitespaceTextNode(nodes[start])) {
    start += 1;
  }

  while (end > start && isWhitespaceTextNode(nodes[end - 1])) {
    end -= 1;
  }

  return nodes.slice(start, end);
}

function isParagraphRunNode(node: any): boolean {
  if (node?.type === "text") {
    return /\S/.test(String(node.value ?? ""));
  }

  if (node?.type !== "element") {
    return false;
  }

  return !BLOCK_LAYOUT_TAGS.has(String(node.tagName ?? "").toLowerCase());
}

function parseInlineImageSet(source: string, rewriter: (value: string) => string): string {
  return source
    .split(",")
    .map((candidate) => {
      const parts = candidate.trim().split(/\s+/);

      if (parts.length === 0) {
        return candidate;
      }

      parts[0] = rewriter(parts[0]);
      return parts.join(" ");
    })
    .join(", ");
}

function normalizeLegacyCenteredDivs(content: string): string {
  return content.replace(/<div\b([^>]*)>/gi, (match, rawAttributes: string) => {
    if (!/\balign\s*=\s*(['"]?)center\1/i.test(rawAttributes)) {
      return match;
    }

    let normalizedAttributes = rawAttributes.replace(/\s*\balign\s*=\s*(['"]?)center\1/gi, "");

    if (/\bclass\s*=\s*(['"])([^'"]*)\1/i.test(normalizedAttributes)) {
      normalizedAttributes = normalizedAttributes.replace(/\bclass\s*=\s*(['"])([^'"]*)\1/i, (_fullMatch, quote: string, value: string) => {
        const classNames = [...new Set(`${value} legacy-center-block`.split(/\s+/).map((item) => item.trim()).filter(Boolean))];
        return ` class=${quote}${classNames.join(" ")}${quote}`;
      });
    } else {
      normalizedAttributes = `${normalizedAttributes} class="legacy-center-block"`;
    }

    return `<div${normalizedAttributes}>`;
  });
}

function normalizeMarkdownSource(content: string): string {
  return normalizeLegacyCenteredDivs(content).replace(/\((\s*):(\d{2,5})(\s*)\)/g, "($1\\:$2$3)");
}

function directoryHasMarkdown(directoryPath: string): boolean {
  try {
    const entries = fs.readdirSync(directoryPath, { withFileTypes: true });

    for (const entry of entries) {
      const absolutePath = path.join(directoryPath, entry.name);

      if (entry.isFile() && /\.md$/i.test(entry.name)) {
        return true;
      }

      if (entry.isDirectory() && directoryHasMarkdown(absolutePath)) {
        return true;
      }
    }
  } catch {
    return false;
  }

  return false;
}

function resolveDirectoryHref(absoluteTarget: string): string {
  const readmePath = path.join(absoluteTarget, "README.md");

  if (fs.existsSync(readmePath)) {
    return relativeFilePathToDocRoutePath(toPosixPath(path.relative(CONTENT_ROOT, readmePath)));
  }

  if (directoryHasMarkdown(absoluteTarget)) {
    return relativeDirectoryPathToFolderRoutePath(toPosixPath(path.relative(CONTENT_ROOT, absoluteTarget)));
  }

  return relativeAssetPathToRoutePath(toPosixPath(path.relative(CONTENT_ROOT, absoluteTarget)));
}

function resolveLocalUrl(relativeDocumentPath: string, rawUrl: string, attributeName: string): string {
  if (
    !rawUrl ||
    rawUrl.startsWith("/") ||
    rawUrl.startsWith("#") ||
    rawUrl.startsWith("mailto:") ||
    rawUrl.startsWith("tel:") ||
    rawUrl.startsWith("data:") ||
    /^[a-z]+:/i.test(rawUrl) ||
    rawUrl.startsWith("//")
  ) {
    return rawUrl;
  }

  const hashIndex = rawUrl.indexOf("#");
  const hash = hashIndex >= 0 ? rawUrl.slice(hashIndex) : "";
  const withoutHash = hashIndex >= 0 ? rawUrl.slice(0, hashIndex) : rawUrl;
  const searchIndex = withoutHash.indexOf("?");
  const search = searchIndex >= 0 ? withoutHash.slice(searchIndex) : "";
  const pathname = searchIndex >= 0 ? withoutHash.slice(0, searchIndex) : withoutHash;
  const absoluteTarget = path.resolve(path.join(CONTENT_ROOT, path.dirname(relativeDocumentPath)), decodeURIComponent(pathname));
  const markdownSiblingPath = `${absoluteTarget}.md`;

  if (!isWithinContentRoot(absoluteTarget)) {
    return rawUrl;
  }

  if (attributeName === "href") {
    if (fs.existsSync(absoluteTarget) && fs.statSync(absoluteTarget).isDirectory()) {
      return `${resolveDirectoryHref(absoluteTarget)}${search}${hash}`;
    }

    if (/\.md$/i.test(absoluteTarget) && fs.existsSync(absoluteTarget)) {
      const relativeTarget = toPosixPath(path.relative(CONTENT_ROOT, absoluteTarget));
      return `${relativeFilePathToDocRoutePath(relativeTarget)}${search}${hash}`;
    }

    if (/\.pdf$/i.test(absoluteTarget) && fs.existsSync(absoluteTarget)) {
      const relativeTarget = toPosixPath(path.relative(CONTENT_ROOT, absoluteTarget));
      return `${relativeFilePathToDocRoutePath(relativeTarget)}${search}${hash}`;
    }

    if (!path.extname(absoluteTarget) && fs.existsSync(markdownSiblingPath)) {
      const relativeTarget = toPosixPath(path.relative(CONTENT_ROOT, markdownSiblingPath));
      return `${relativeFilePathToDocRoutePath(relativeTarget)}${search}${hash}`;
    }
  }

  const relativeTarget = toPosixPath(path.relative(CONTENT_ROOT, absoluteTarget));
  return `${relativeAssetPathToRoutePath(relativeTarget)}${search}${hash}`;
}

function remarkDirectiveRenderer() {
  return (tree: MdastRoot) => {
    visit(tree, (node: any, index: number | undefined, parent: any) => {
      if (!["containerDirective", "leafDirective", "textDirective"].includes(node.type)) {
        return;
      }

      const name = String(node.name ?? "div").toLowerCase();
      const isNameRenderableDirective = /^[a-z][a-z0-9-]*$/i.test(name);

      if (!isNameRenderableDirective && parent && typeof index === "number") {
        parent.children[index] = {
          type: "text",
          value: `:${String(node.name ?? "").trim()}`,
        };
        return false;
      }

      const isInline = node.type === "textDirective";
      const className = ["directive", `directive-${name}`];
      const properties: Record<string, unknown> = {};
      const attributes = node.attributes ?? {};

      for (const [key, value] of Object.entries(attributes)) {
        if (key === "class" || key === "className") {
          className.push(
            ...String(value)
              .split(/\s+/)
              .map((item) => item.trim())
              .filter(Boolean),
          );
        } else {
          properties[key] = value;
        }
      }

      if (name === "mark" || name === "highlight") {
        properties.className = className;
        node.data = { hName: "mark", hProperties: properties };
        return false;
      }

      if (name === "kbd") {
        properties.className = className;
        node.data = { hName: "kbd", hProperties: properties };
        return false;
      }

      if (name === "badge" || name === "pill" || name === "tag") {
        properties.className = [...className, "inline-badge"];
        node.data = { hName: "span", hProperties: properties };
        return false;
      }

      const tagName = CALLOUT_TAGS.has(name)
        ? "aside"
        : SEMANTIC_DIRECTIVE_TAGS.has(name)
          ? name
          : isInline
            ? "span"
            : "div";

      if (CALLOUT_TAGS.has(name)) {
        properties["data-callout-type"] = name;
        properties["data-callout-title"] = String(attributes.title ?? CALLOUT_TITLES[name] ?? name);
        className.push("callout", `callout-${name}`);
      }

      properties.className = className;
      node.data = { hName: tagName, hProperties: properties };
      return false;
    });
  };
}

function remarkCallouts() {
  return (tree: MdastRoot) => {
    visit(tree, "blockquote", (node: any) => {
      const firstParagraph = node.children?.[0];

      if (!firstParagraph || firstParagraph.type !== "paragraph") {
        return;
      }

      const firstText = firstParagraph.children?.find((child: any) => child.type === "text");

      if (!firstText || typeof firstText.value !== "string") {
        return;
      }

      const match = firstText.value.match(/^\s*\[!(NOTE|INFO|WARNING|TIP|IMPORTANT|CAUTION|SUCCESS|ERROR)\]\s*(.*)$/i);

      if (!match) {
        return;
      }

      const type = match[1].toLowerCase();
      const title = match[2].trim() || CALLOUT_TITLES[type];
      firstText.value = firstText.value.replace(match[0], "").trimStart();

      if (!firstText.value) {
        firstParagraph.children = firstParagraph.children.filter((child: any) => child !== firstText);
      }

      if (firstParagraph.children.length === 0) {
        node.children.shift();
      }

      node.data = {
        hName: "aside",
        hProperties: {
          className: ["callout", `callout-${type}`],
          "data-callout-type": type,
          "data-callout-title": title,
        },
      };
    });
  };
}

function remarkMermaidBlocks() {
  return (tree: MdastRoot) => {
    visit(tree, "code", (node: any) => {
      if (String(node.lang ?? "").toLowerCase() !== "mermaid") {
        return;
      }

      node.data = {
        hName: "div",
        hProperties: {
          className: ["mermaid-shell"],
        },
        hChildren: [
          {
            type: "element",
            tagName: "div",
            properties: {
              className: ["mermaid"],
            },
            children: [
              {
                type: "text",
                value: node.value,
              },
            ],
          },
        ],
      };
    });
  };
}

function rehypeWrapTables() {
  return (tree: HastRoot) => {
    visit(tree, "element", (node: any, index: number | undefined, parent: any) => {
      if (!parent || typeof index !== "number" || node.tagName !== "table") {
        return;
      }

      if (parent.tagName === "div" && parent.properties?.className?.includes?.("table-wrap")) {
        return;
      }

      parent.children[index] = {
        type: "element",
        tagName: "div",
        properties: {
          className: ["table-wrap"],
        },
        children: [node],
      };
    });
  };
}

function rehypeNormalizeLegacyCenteredBlocks() {
  return (tree: HastRoot) => {
    visit(tree, "element", (node: any, index: number | undefined, parent: any) => {
      if (node.tagName !== "div") {
        return;
      }

      const align = String(node.properties?.align ?? "").toLowerCase();

      if (align !== "center") {
        return;
      }

      const children = trimNodeRun(Array.isArray(node.children) ? node.children : []);

      delete node.properties.align;

      if (
        parent &&
        typeof index === "number" &&
        children.length === 1 &&
        children[0]?.type === "element" &&
        /^h[1-6]$/.test(String(children[0].tagName ?? ""))
      ) {
        parent.children[index] = children[0];
        return false;
      }

      node.properties.className = [...new Set([...getClassNames(node.properties), "legacy-center-block"])];
      return false;
    });
  };
}

function rehypeWrapLooseParagraphRuns() {
  return (tree: HastRoot) => {
    visit(tree, (node: any) => {
      if (!Array.isArray(node.children)) {
        return false;
      }

      if (node.type === "element") {
        const tagName = String(node.tagName ?? "").toLowerCase();
        const classNames = getClassNames(node.properties);

        if (
          BLOCK_LAYOUT_TAGS.has(tagName) &&
          tagName !== "article" &&
          tagName !== "div" &&
          tagName !== "section" &&
          tagName !== "aside" &&
          tagName !== "main"
        ) {
          return false;
        }

        if (classNames.includes("table-wrap") || classNames.includes("legacy-center-block")) {
          return false;
        }
      }

      const nextChildren: any[] = [];
      let paragraphRun: any[] = [];

      const flushParagraphRun = () => {
        if (paragraphRun.length === 0) {
          return;
        }

        const trimmedRun = trimNodeRun(paragraphRun);

        if (trimmedRun.length > 0) {
          nextChildren.push({
            type: "element",
            tagName: "p",
            properties: {},
            children: trimmedRun,
          });
        }

        paragraphRun = [];
      };

      for (const child of node.children) {
        if (isParagraphRunNode(child)) {
          paragraphRun.push(child);
          continue;
        }

        flushParagraphRun();
        nextChildren.push(child);
      }

      flushParagraphRun();
      node.children = nextChildren;
      return false;
    });
  };
}

function rehypeCollectHeadings(headings: DocumentHeading[]) {
  return (tree: HastRoot) => {
    visit(tree, "element", (node: any) => {
      if (!/^h[1-6]$/.test(node.tagName)) {
        return;
      }

      const id = typeof node.properties?.id === "string" ? node.properties.id : "";
      const text = extractHastText(node).replace(/\s+/g, " ").trim();

      if (!id || !text) {
        return;
      }

      headings.push({
        id,
        text,
        depth: Number(node.tagName.slice(1)),
      });
    });
  };
}

function rehypeRewriteUrls(relativeDocumentPath: string) {
  return (tree: HastRoot) => {
    visit(tree, "element", (node: any) => {
      const properties = node.properties ?? {};

      for (const attributeName of ["href", "src", "poster", "data", "xlinkHref"]) {
        if (typeof properties[attributeName] === "string") {
          properties[attributeName] = resolveLocalUrl(relativeDocumentPath, properties[attributeName], attributeName);
        }
      }

      for (const srcSetAttribute of ["srcset", "srcSet", "imagesrcset", "imageSrcSet"]) {
        if (typeof properties[srcSetAttribute] === "string") {
          properties[srcSetAttribute] = parseInlineImageSet(properties[srcSetAttribute], (value) =>
            resolveLocalUrl(relativeDocumentPath, value, "src"),
          );
        }
      }

      if (node.tagName === "a" && typeof properties.href === "string" && /^(https?:)?\/\//i.test(properties.href)) {
        properties.rel = "noreferrer";
        properties.target = "_blank";
      }

      node.properties = properties;
    });
  };
}

export async function renderMarkdownDocument(
  content: string,
  relativeDocumentPath: string,
): Promise<{ html: string; headings: DocumentHeading[] }> {
  const headings: DocumentHeading[] = [];
  const normalizedContent = normalizeMarkdownSource(content);

  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkDirective)
    .use(remarkDirectiveRenderer)
    .use(remarkCallouts)
    .use(remarkMermaidBlocks)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeNormalizeLegacyCenteredBlocks)
    .use(rehypeWrapTables)
    .use(rehypeWrapLooseParagraphRuns)
    .use(rehypeKatex, {
      strict(errorCode) {
        return errorCode === "unknownSymbol" ? "ignore" : "warn";
      },
    })
    .use(rehypeSlug)
    .use(rehypeCollectHeadings, headings)
    .use(rehypeRewriteUrls, relativeDocumentPath)
    .use(rehypeAutolinkHeadings, {
      behavior: "append",
      properties: {
        className: ["heading-anchor"],
        ariaLabel: "Anchor link",
      },
      content: {
        type: "text",
        value: "#",
      },
    })
    .use(rehypePrettyCode, {
      theme: "github-light",
      keepBackground: false,
      defaultLang: {
        block: "text",
        inline: "text",
      },
      onVisitHighlightedLine(node) {
        node.properties.className = [...((node.properties.className as string[]) ?? []), "highlighted-line"];
      },
      onVisitHighlightedChars(node) {
        node.properties.className = [...((node.properties.className as string[]) ?? []), "highlighted-chars"];
      },
    })
    .use(rehypeStringify)
    .process(normalizedContent);

  return {
    html: String(file),
    headings,
  };
}
