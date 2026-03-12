#!/usr/bin/env python3
"""Normalize local asset references in Markdown and HTML content.

This script is intentionally generalized:
* It indexes every file beneath one or more asset roots.
* It scans one or more content roots recursively for Markdown/HTML files.
* It resolves local image references even when they are written as absolute,
  relative, or partially prefixed paths.
* It rewrites those references into a canonical form that remains valid when a
  Jekyll site is served from a non-root base URL.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".html", ".htm"}
FRONT_MATTER_IMAGE_KEYS = {
    "image",
    "thumbnail",
    "banner",
    "cover",
    "hero_image",
    "header_image",
    "og_image",
}
EXTERNAL_PREFIXES = ("http://", "https://", "data:", "mailto:", "tel:", "#")

MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<url><[^>]+>|\{\{[^)]*\}\}|[^)\s]+)(?P<tail>(?:\s+\"[^\"]*\")?)\)"
)
HTML_ASSET_ATTRIBUTE_PATTERN = re.compile(
    r"(?P<prefix><(?P<tag>[a-z0-9:_-]+)\b[^>]*?\b(?P<attr>src|poster)=)(?P<quote>\\?[\"'])(?P<url>.*?)(?P=quote)",
    re.IGNORECASE,
)
FRONT_MATTER_KEY_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_-]+)\s*:\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)
FRONT_MATTER_START_PATTERN = re.compile(r"\A---[ \t]*\r?\n")
FRONT_MATTER_END_PATTERN = re.compile(r"^---[ \t]*$", re.MULTILINE)
LIQUID_RELATIVE_URL_PATTERN = re.compile(
    r"^\{\{\s*['\"](?P<path>/[^'\"]+)['\"]\s*\|\s*relative_url\s*\}\}$"
)
MARKDOWN_REFERENCE_PATTERN = re.compile(
    r"(?P<label>\[[^\]]+\]):\s*(?P<url><[^>]+>|\{\{[^\\n]*\}\}|[^\s]+)",
)


def iter_content_files(content_roots: Iterable[Path]) -> Iterable[Path]:
    for content_root in content_roots:
        for path in sorted(content_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def split_query_fragment(value: str) -> tuple[str, str]:
    query_index = value.find("?")
    fragment_index = value.find("#")
    indices = [index for index in (query_index, fragment_index) if index >= 0]
    if not indices:
        return value, ""
    split_at = min(indices)
    return value[:split_at], value[split_at:]


def clean_reference(raw_value: str) -> str:
    value = raw_value.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    liquid_match = LIQUID_RELATIVE_URL_PATTERN.match(value)
    if liquid_match is not None:
        value = liquid_match.group("path")
    return value


def normalize_rel_path(path: Path, site_root: Path) -> str:
    return path.relative_to(site_root).as_posix()


def normalize_ref_path(raw_path: str) -> str:
    value = raw_path.replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    if value.startswith("/"):
        value = value[1:]
    normalized = str(PurePosixPath(value))
    return "" if normalized == "." else normalized


def is_front_matter_asset_key(key: str) -> bool:
    normalized = key.lower()
    if normalized in FRONT_MATTER_IMAGE_KEYS:
        return True
    return normalized.endswith(("image", "img", "_image", "-image", "_img", "-img"))


@dataclass(frozen=True)
class AssetRecord:
    relative_path: str
    absolute_path: Path


class AssetIndex:
    def __init__(self, site_root: Path, asset_roots: list[Path]) -> None:
        self.site_root = site_root
        self.asset_roots = asset_roots
        self.by_relative_path: dict[str, AssetRecord] = {}
        self.by_basename: dict[str, list[AssetRecord]] = defaultdict(list)
        self.by_suffix: dict[str, list[AssetRecord]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        for asset_root in self.asset_roots:
            for file_path in sorted(asset_root.rglob("*")):
                if not file_path.is_file():
                    continue
                relative_path = normalize_rel_path(file_path, self.site_root)
                normalized_key = relative_path.lower()
                record = AssetRecord(relative_path=relative_path, absolute_path=file_path)
                self.by_relative_path[normalized_key] = record
                self.by_basename[file_path.name.lower()].append(record)
                parts = relative_path.split("/")
                for start in range(1, len(parts)):
                    suffix = "/".join(parts[start:]).lower()
                    self.by_suffix[suffix].append(record)

    def resolve(self, ref: str, file_path: Path) -> AssetRecord | None:
        clean_value = clean_reference(ref)
        if not clean_value or clean_value.startswith(EXTERNAL_PREFIXES) or "{{" in clean_value:
            return None

        path_part, _suffix = split_query_fragment(clean_value)
        normalized_ref = normalize_ref_path(path_part)
        if not normalized_ref:
            return None

        direct = self.by_relative_path.get(normalized_ref.lower())
        if direct is not None:
            return direct

        if "assets/" in normalized_ref.lower():
            asset_suffix = normalized_ref.lower().split("assets/", 1)[1]
            suffix_matches = self.by_suffix.get(asset_suffix)
            if suffix_matches and len(suffix_matches) == 1:
                return suffix_matches[0]

        candidate_from_file = self._resolve_relative_to_file(normalized_ref, file_path)
        if candidate_from_file is not None:
            return candidate_from_file

        basename_matches = self.by_basename.get(PurePosixPath(normalized_ref).name.lower())
        if basename_matches and len(basename_matches) == 1:
            return basename_matches[0]

        suffix_matches = self.by_suffix.get(normalized_ref.lower())
        if suffix_matches and len(suffix_matches) == 1:
            return suffix_matches[0]

        return None

    def _resolve_relative_to_file(self, normalized_ref: str, file_path: Path) -> AssetRecord | None:
        try:
            candidate = (file_path.parent / Path(normalized_ref)).resolve()
            relative = candidate.relative_to(self.site_root.resolve()).as_posix().lower()
        except Exception:
            return None
        return self.by_relative_path.get(relative)


def discover_asset_roots(site_root: Path, asset_dir_names: list[str]) -> list[Path]:
    discovered: list[Path] = []
    wanted = {name.lower() for name in asset_dir_names}
    for path in sorted(site_root.rglob("*")):
        if path.is_dir() and path.name.lower() in wanted:
            discovered.append(path)
    return discovered


def canonical_jekyll_url(relative_path: str) -> str:
    return "{{ '" + "/" + relative_path + "' | relative_url }}"


def canonical_relative_url(asset_path: Path, file_path: Path) -> str:
    relative_path = os.path.relpath(asset_path, start=file_path.parent)
    return PurePosixPath(relative_path.replace("\\", "/")).as_posix()


def build_output_url(mode: str, asset: AssetRecord, file_path: Path) -> str:
    if mode == "jekyll":
        return canonical_jekyll_url(asset.relative_path)
    if mode == "relative":
        return canonical_relative_url(asset.absolute_path, file_path)
    raise ValueError(f"Unsupported mode: {mode}")


def normalize_markdown_images(
    text: str,
    file_path: Path,
    asset_index: AssetIndex,
    output_mode: str,
) -> tuple[str, int]:
    changes = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal changes
        url_token = match.group("url")
        raw_url = clean_reference(url_token)
        asset = asset_index.resolve(raw_url, file_path)
        if asset is None:
            return match.group(0)
        changes += 1
        return f"![{match.group('alt')}]({build_output_url(output_mode, asset, file_path)}{match.group('tail')})"

    return MARKDOWN_IMAGE_PATTERN.sub(replacer, text), changes


def normalize_html_asset_attributes(
    text: str,
    file_path: Path,
    asset_index: AssetIndex,
    output_mode: str,
) -> tuple[str, int]:
    changes = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal changes
        raw_url = clean_reference(match.group("url"))
        asset = asset_index.resolve(raw_url, file_path)
        if asset is None:
            return match.group(0)
        changes += 1
        return f"{match.group('prefix')}{match.group('quote')}{build_output_url(output_mode, asset, file_path)}{match.group('quote')}"

    return HTML_ASSET_ATTRIBUTE_PATTERN.sub(replacer, text), changes


def normalize_markdown_reference_definitions(
    text: str,
    file_path: Path,
    asset_index: AssetIndex,
    output_mode: str,
) -> tuple[str, int]:
    changes = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal changes
        raw_url = clean_reference(match.group("url"))
        asset = asset_index.resolve(raw_url, file_path)
        if asset is None:
            return match.group(0)
        changes += 1
        return f"{match.group('label')}: {build_output_url(output_mode, asset, file_path)}"

    return MARKDOWN_REFERENCE_PATTERN.sub(replacer, text), changes


def normalize_front_matter_keys(
    text: str,
    file_path: Path,
    asset_index: AssetIndex,
    output_mode: str,
) -> tuple[str, int]:
    start_match = FRONT_MATTER_START_PATTERN.match(text)
    if start_match is None:
        return text, 0

    closing_match = FRONT_MATTER_END_PATTERN.search(text, start_match.end())
    if closing_match is None:
        return text, 0

    front_matter_end = closing_match.end()
    front_matter = text[:front_matter_end]
    body = text[front_matter_end:]
    changes = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal changes
        key = match.group("key")
        if not is_front_matter_asset_key(key):
            return match.group(0)

        raw_value = match.group("value").strip()
        if raw_value.startswith(("'", '"')) and raw_value.endswith(raw_value[0]):
            unwrapped = raw_value[1:-1]
        else:
            unwrapped = raw_value

        asset = asset_index.resolve(unwrapped, file_path)
        if asset is None:
            return match.group(0)

        changes += 1
        return f'{match.group("indent")}{key}: "{build_output_url(output_mode, asset, file_path)}"'

    normalized_front_matter = FRONT_MATTER_KEY_PATTERN.sub(replacer, front_matter)
    return normalized_front_matter + body, changes


def process_file(
    file_path: Path,
    asset_index: AssetIndex,
    write: bool,
    rewrite_front_matter: bool,
    output_mode: str,
) -> tuple[int, bool]:
    original_text = file_path.read_text(encoding="utf-8", errors="surrogateescape")
    updated_text = original_text
    total_changes = 0

    if rewrite_front_matter:
        updated_text, changes = normalize_front_matter_keys(
            updated_text,
            file_path,
            asset_index,
            output_mode,
        )
        total_changes += changes

    updated_text, changes = normalize_html_asset_attributes(
        updated_text,
        file_path,
        asset_index,
        output_mode,
    )
    total_changes += changes

    updated_text, changes = normalize_markdown_reference_definitions(
        updated_text,
        file_path,
        asset_index,
        output_mode,
    )
    total_changes += changes

    updated_text, changes = normalize_markdown_images(
        updated_text,
        file_path,
        asset_index,
        output_mode,
    )
    total_changes += changes

    if updated_text == original_text:
        return 0, False

    if write:
        file_path.write_text(updated_text, encoding="utf-8", errors="surrogateescape")

    return total_changes, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize local image references so asset URLs stay valid behind a Jekyll base URL."
    )
    parser.add_argument(
        "site_root",
        nargs="?",
        default=".",
        help="Site root that contains content and asset folders. Defaults to the current directory.",
    )
    parser.add_argument(
        "--content-root",
        action="append",
        default=[],
        help="Content root to scan recursively. May be repeated. Defaults to the site root.",
    )
    parser.add_argument(
        "--asset-dir-name",
        action="append",
        default=["assets"],
        help="Directory name to treat as an asset root. May be repeated. Defaults to 'assets'.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist rewritten files. Without this flag the script performs a dry run.",
    )
    parser.add_argument(
        "--rewrite-front-matter",
        action="store_true",
        help="Also rewrite front matter image-like keys. Disabled by default because layouts may already normalize them.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["relative", "jekyll"],
        default="relative",
        help="Rewrite targets as preview-safe relative paths or Jekyll Liquid relative_url expressions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    site_root = Path(args.site_root).resolve()
    content_roots = [Path(path).resolve() for path in args.content_root] if args.content_root else [site_root]
    asset_roots = discover_asset_roots(site_root, args.asset_dir_name)

    if not asset_roots:
        print("No asset roots were found.", file=sys.stderr)
        return 1

    asset_index = AssetIndex(site_root=site_root, asset_roots=asset_roots)

    files_seen = 0
    files_changed = 0
    total_rewrites = 0

    for file_path in iter_content_files(content_roots):
        files_seen += 1
        change_count, changed = process_file(
            file_path,
            asset_index,
            write=args.write,
            rewrite_front_matter=args.rewrite_front_matter,
            output_mode=args.output_mode,
        )
        total_rewrites += change_count
        if changed:
            files_changed += 1
            print(f"{'UPDATED' if args.write else 'WOULD-UPDATE'} {file_path.relative_to(site_root).as_posix()} ({change_count} rewrites)")

    mode = "write" if args.write else "dry-run"
    print(
        f"Completed {mode}: scanned={files_seen} changed={files_changed} rewrites={total_rewrites} asset_roots={len(asset_roots)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
