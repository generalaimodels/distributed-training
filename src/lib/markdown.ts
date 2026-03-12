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
    visit(tree, (node: any) => {
      if (!["containerDirective", "leafDirective", "textDirective"].includes(node.type)) {
        return false;
      }

      const name = String(node.name ?? "div").toLowerCase();
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
    .use(rehypeWrapTables)
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
    .process(content);

  return {
    html: String(file),
    headings,
  };
}
