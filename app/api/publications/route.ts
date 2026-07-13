import { publications } from "../../data/publications";

type OpenAlexWork = {
  cited_by_count?: number;
  updated_date?: string;
};

export async function GET() {
  const syncedAt = new Date().toISOString();

  const results = await Promise.all(
    publications.map(async (publication) => {
      try {
        const response = await fetch(
          `https://api.openalex.org/works/https://doi.org/${publication.doi}?mailto=zhangst@stu.xjtu.edu.cn`,
          {
            headers: {
              Accept: "application/json",
              "User-Agent": "Shengtao-Zhang-Academic-Homepage/1.0",
            },
          },
        );
        if (!response.ok) throw new Error(`OpenAlex returned ${response.status}`);
        const work = (await response.json()) as OpenAlexWork;
        return {
          title: publication.title,
          venue: publication.venue,
          year: publication.year,
          paperUrl: publication.paperUrl,
          doi: publication.doi,
          citationCount: Number(work.cited_by_count) || 0,
          sourceUpdatedAt: work.updated_date || null,
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
        };
      }
    }),
  );

  const liveCount = results.filter(
    (publication) => publication.citationCount !== null,
  ).length;

  return Response.json(
    {
      source: liveCount > 0 ? "OpenAlex" : "verified-local",
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
