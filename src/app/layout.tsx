import type { Metadata } from "next";
import { Space_Grotesk, Source_Serif_4 } from "next/font/google";
import type { ReactNode } from "react";

import "katex/dist/katex.min.css";

import { ReaderLayoutStateProvider } from "@/components/reader-layout-state";
import { ReaderStateProvider } from "@/components/reader-state";
import { SiteHeader } from "@/components/site-header";

import "./globals.css";

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
});
const bodyFont = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: {
    default: "Distribution Training",
    template: "%s | Distribution Training",
  },
  description:
    "Advanced Markdown and raw HTML technical content rendered as a polished documentation and knowledge website.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={`${displayFont.variable} ${bodyFont.variable}`}>
        <ReaderLayoutStateProvider>
          <ReaderStateProvider>
            <div className="site-orb site-orb-left" />
            <div className="site-orb site-orb-right" />
            <SiteHeader />
            <main className="site-main">{children}</main>
            <footer className="site-footer">
              <div className="container footer-inner">
                <p>Markdown, math, code, media, callouts, raw HTML, and future content folders all flow through one TypeScript renderer.</p>
              </div>
            </footer>
          </ReaderStateProvider>
        </ReaderLayoutStateProvider>
      </body>
    </html>
  );
}
