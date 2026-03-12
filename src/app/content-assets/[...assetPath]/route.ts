import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

import { contentType, lookup } from "mime-types";
import { NextResponse } from "next/server";

import { CONTENT_ROOT, encodePathSegments, humanizeToken, isWithinContentRoot } from "@/lib/pathing";

interface ContentAssetRouteProps {
  params: Promise<{
    assetPath: string[];
  }>;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export async function GET(_: Request, { params }: ContentAssetRouteProps) {
  const { assetPath } = await params;
  const absolutePath = path.resolve(CONTENT_ROOT, ...assetPath.map((segment) => decodeURIComponent(segment)));

  if (!isWithinContentRoot(absolutePath)) {
    return new NextResponse("Not found", { status: 404 });
  }

  try {
    const targetStats = await stat(absolutePath);

    if (targetStats.isDirectory()) {
      const entries = await readdir(absolutePath, { withFileTypes: true });
      const currentPath = assetPath.join("/") || "content root";
      const listItems = entries
        .sort((left, right) => {
          if (left.isDirectory() !== right.isDirectory()) {
            return left.isDirectory() ? -1 : 1;
          }

          return left.name.localeCompare(right.name);
        })
        .map((entry) => {
          const href = `/content-assets/${encodePathSegments([...assetPath, entry.name])}`;
          const suffix = entry.isDirectory() ? "/" : "";
          return `<li><a href="${href}">${escapeHtml(entry.name)}${suffix}</a><span>${escapeHtml(humanizeToken(entry.name))}</span></li>`;
        })
        .join("");

      const parentHref =
        assetPath.length > 1 ? `/content-assets/${encodePathSegments(assetPath.slice(0, -1))}` : assetPath.length === 1 ? "/docs" : "/";
      const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Asset Directory - ${escapeHtml(currentPath)}</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #f7f4ec; color: #13243d; }
      main { max-width: 880px; margin: 0 auto; padding: 32px 16px 56px; }
      a { color: #0f7c72; text-decoration: none; }
      ul { list-style: none; padding: 0; margin: 24px 0 0; display: grid; gap: 12px; }
      li { display: flex; justify-content: space-between; gap: 16px; padding: 14px 16px; border-radius: 16px; background: rgba(255,255,255,0.86); border: 1px solid rgba(19,36,61,0.1); }
      h1 { margin: 12px 0 6px; font-size: 32px; }
      p { color: #556578; }
      .back { display: inline-flex; margin-bottom: 8px; font-weight: 700; }
      span { color: #556578; }
    </style>
  </head>
  <body>
    <main>
      <a class="back" href="${parentHref}">Back</a>
      <h1>${escapeHtml(currentPath)}</h1>
      <p>This directory contains repository assets referenced by the rendered Markdown content.</p>
      <ul>${listItems}</ul>
    </main>
  </body>
</html>`;

      return new NextResponse(html, {
        headers: {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "public, max-age=3600",
        },
      });
    }

    const file = await readFile(absolutePath);
    const mime = contentType(lookup(absolutePath) || "application/octet-stream") || "application/octet-stream";

    return new NextResponse(file, {
      headers: {
        "Content-Type": mime,
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch {
    return new NextResponse("Not found", { status: 404 });
  }
}
