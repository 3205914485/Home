import { publications } from "../../data/publications";

type OpenAlexWork = {
  cited_by_count?: number;
  updated_date?: string;
};

type CrossrefResponse = {
  message?: {
    "is-referenced-by-count"?: number;
    indexed?: { "date-time"?: string };
  };
};

type DataCiteResponse = {
  data?: {
    attributes?: {
      citationCount?: number;
      updated?: string;
    };
  };
};

type CitationSignal = {
  citationCount: number;
  sourceUpdatedAt: string | null;
  citationSource: "Crossref" | "DataCite" | "OpenAlex";
};

async function fetchCrossref(doi: string): Promise<CitationSignal | null> {
  const response = await fetch(
    `https://api.crossref.org/works/${encodeURIComponent(doi)}?mailto=zhangst@stu.xjtu.edu.cn`,
    { headers: { Accept: "application/json" } },
  );
  if (!response.ok) return null;
  const payload = (await response.json()) as CrossrefResponse;
  const work = payload.message;
  if (typeof work?.["is-referenced-by-count"] !== "number") return null;
  return {
    citationCount: work["is-referenced-by-count"],
    sourceUpdatedAt: work.indexed?.["date-time"] || null,
    citationSource: "Crossref",
  };
}

async function fetchDataCite(doi: string): Promise<CitationSignal | null> {
  const response = await fetch(
    `https://api.datacite.org/dois/${encodeURIComponent(doi)}`,
    { headers: { Accept: "application/vnd.api+json" } },
  );
  if (!response.ok) return null;
  const payload = (await response.json()) as DataCiteResponse;
  const work = payload.data?.attributes;
  if (typeof work?.citationCount !== "number") return null;
  return {
    citationCount: work.citationCount,
    sourceUpdatedAt: work.updated || null,
    citationSource: "DataCite",
  };
}

async function fetchOpenAlex(doi: string): Promise<CitationSignal | null> {
  const response = await fetch(
    `https://api.openalex.org/works/https://doi.org/${doi}?mailto=zhangst@stu.xjtu.edu.cn`,
    {
      headers: {
        Accept: "application/json",
        "User-Agent": "Shengtao-Zhang-Academic-Homepage/1.0",
      },
    },
  );
  if (!response.ok) return null;
  const work = (await response.json()) as OpenAlexWork;
  if (typeof work.cited_by_count !== "number") return null;
  return {
    citationCount: work.cited_by_count,
    sourceUpdatedAt: work.updated_date || null,
    citationSource: "OpenAlex",
  };
}

async function safelyFetch(
  fetcher: (doi: string) => Promise<CitationSignal | null>,
  doi: string,
) {
  try {
    return await fetcher(doi);
  } catch {
    return null;
  }
}

export async function GET() {
  const syncedAt = new Date().toISOString();

  const results = await Promise.all(
    publications.map(async (publication) => {
      try {
        const preferredRegistry = publication.doi.startsWith("10.48550/")
          ? fetchDataCite
          : fetchCrossref;
        const signal =
          (await safelyFetch(preferredRegistry, publication.doi)) ||
          (await safelyFetch(fetchOpenAlex, publication.doi));
        return {
          title: publication.title,
          venue: publication.venue,
          year: publication.year,
          paperUrl: publication.paperUrl,
          doi: publication.doi,
          citationCount: signal?.citationCount ?? null,
          sourceUpdatedAt: signal?.sourceUpdatedAt ?? null,
          citationSource: signal?.citationSource ?? null,
        };
      } catch {
        return {
          title: publication.title,
          venue: publication.venue,
          year: publication.year,
          paperUrl: publication.paperUrl,
          doi: publication.doi,
          citationCount: null,
          sourceUpdatedAt: null,
          citationSource: null,
        };
      }
    }),
  );

  const liveSources = Array.from(
    new Set(
      results.flatMap((publication) =>
        publication.citationSource ? [publication.citationSource] : [],
      ),
    ),
  );

  return Response.json(
    {
      source: liveSources.length > 0 ? liveSources.join(" + ") : "verified-local",
      syncedAt,
      publications: results,
    },
    {
      headers: {
        "Cache-Control":
          "public, max-age=0, s-maxage=21600, stale-while-revalidate=86400",
      },
    },
  );
}
