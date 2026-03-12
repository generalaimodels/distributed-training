"use client";

import { useEffect } from "react";

export function MermaidLoader() {
  useEffect(() => {
    let active = true;

    async function start() {
      const { default: mermaid } = await import("mermaid");

      if (!active) {
        return;
      }

      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: "neutral",
      });

      await mermaid.run({
        nodes: document.querySelectorAll(".mermaid"),
      });
    }

    void start();

    return () => {
      active = false;
    };
  }, []);

  return null;
}
