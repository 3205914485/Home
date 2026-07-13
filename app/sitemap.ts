import type { MetadataRoute } from "next";
import { publications } from "./data/publications";
import { getSiteOrigin } from "./lib/site-url";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const siteUrl = await getSiteOrigin();
  return [
    {
      url: `${siteUrl}/`,
      lastModified: new Date("2026-07-13"),
      changeFrequency: "weekly",
      priority: 1,
    },
    ...publications.map((publication) => ({
      url: `${siteUrl}/publications/${publication.slug}`,
      lastModified: new Date("2026-07-13"),
      changeFrequency: "monthly" as const,
      priority: publication.featured ? 0.9 : 0.75,
    })),
  ];
}
