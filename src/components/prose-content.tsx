"use client";

import { startTransition, useCallback } from "react";
import type { MouseEvent } from "react";

import { useRouter } from "next/navigation";

import { scrollToAnchorId } from "@/lib/reader-scroll";

interface ProseContentProps {
  html: string;
}

const INTERNAL_ROUTE_EXPRESSION = /^\/(?:docs|tags|folders|collections)(?:\/|$)/;

export function ProseContent({ html }: ProseContentProps) {
  const router = useRouter();

  const handleClick = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }

      const target = event.target;

      if (!(target instanceof Element)) {
        return;
      }

      const anchor = target.closest("a");

      if (!(anchor instanceof HTMLAnchorElement)) {
        return;
      }

      const href = anchor.getAttribute("href");

      if (
        !href ||
        anchor.target === "_blank" ||
        anchor.hasAttribute("download")
      ) {
        return;
      }

      if (href.startsWith("#")) {
        const anchorId = decodeURIComponent(href.slice(1));

        if (!anchorId) {
          return;
        }

        event.preventDefault();
        scrollToAnchorId(anchorId);
        return;
      }

      if (!INTERNAL_ROUTE_EXPRESSION.test(href)) {
        return;
      }

      event.preventDefault();
      startTransition(() => {
        router.push(href, { scroll: true });
      });
    },
    [router],
  );

  return <article className="content-prose" onClick={handleClick} dangerouslySetInnerHTML={{ __html: html }} />;
}
