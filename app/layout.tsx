import type { Metadata, Viewport } from "next";
import "./globals.css";
import { getSiteOrigin } from "./lib/site-url";

export async function generateMetadata(): Promise<Metadata> {
  const origin = await getSiteOrigin();
  const socialImage = `${origin}/og.png`;

  return {
    metadataBase: new URL(origin),
    title: {
      default: "Shengtao Zhang — Agents, RL & Memory",
      template: "%s | Shengtao Zhang",
    },
    description:
      "Shengtao Zhang researches self-evolving agents, runtime reinforcement learning, and memory at SJTU-MARL.",
    keywords: [
      "Shengtao Zhang",
      "SJTU-MARL",
      "AI agents",
      "reinforcement learning",
      "agent memory",
      "MemRL",
      "MemQ",
    ],
    authors: [{ name: "Shengtao Zhang" }],
    creator: "Shengtao Zhang",
    alternates: { canonical: origin },
    openGraph: {
      type: "website",
      locale: "en_US",
      url: origin,
      siteName: "Shengtao Zhang",
      title: "Shengtao Zhang — Agents, RL & Memory",
      description: "Agents that remember, adapt, and improve at runtime.",
      images: [
        {
          url: socialImage,
          width: 1200,
          height: 630,
          alt: "Shengtao Zhang — Agents that remember, adapt, and improve at runtime.",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "Shengtao Zhang — Agents, RL & Memory",
      description: "Agents that remember, adapt, and improve at runtime.",
      images: [socialImage],
    },
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        "max-image-preview": "large",
        "max-snippet": -1,
      },
    },
  };
}

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#f5f4ef",
  colorScheme: "light",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
