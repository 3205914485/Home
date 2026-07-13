import type { Metadata } from "next";
import { notFound } from "next/navigation";
import {
  publicationBySlug,
  publications,
} from "../../data/publications";
import { getSiteOrigin } from "../../lib/site-url";

type PageProps = {
  params: Promise<{ slug: string }>;
};

export function generateStaticParams() {
  return publications.map((publication) => ({ slug: publication.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const publication = publicationBySlug.get(slug);
  if (!publication) return {};
  const siteOrigin = await getSiteOrigin();
  const publicationUrl = `${siteOrigin}/publications/${publication.slug}`;

  return {
    title: publication.title,
    description: publication.abstract,
    alternates: {
      canonical: publicationUrl,
    },
    openGraph: {
      type: "article",
      title: publication.title,
      description: publication.contribution,
      url: publicationUrl,
      publishedTime: `${publication.year}-01-01`,
      authors: publication.authors,
    },
    other: {
      citation_title: publication.title,
      citation_author: publication.authors,
      citation_publication_date: String(publication.year),
      citation_online_date: String(publication.year),
      citation_journal_title: publication.venue,
      citation_doi: publication.doi,
      citation_pdf_url: publication.pdfUrl,
      citation_abstract_html_url: publicationUrl,
    },
  };
}

export default async function PublicationPage({ params }: PageProps) {
  const { slug } = await params;
  const publication = publicationBySlug.get(slug);
  if (!publication) notFound();
  const siteOrigin = await getSiteOrigin();

  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "ScholarlyArticle",
    headline: publication.title,
    name: publication.title,
    datePublished: String(publication.year),
    abstract: publication.abstract,
    author: publication.authors.map((name) => ({
      "@type": "Person",
      name,
    })),
    sameAs: publication.paperUrl,
    url: `${siteOrigin}/publications/${publication.slug}`,
    identifier: [
      { "@type": "PropertyValue", propertyID: "DOI", value: publication.doi },
      { "@type": "PropertyValue", propertyID: "arXiv", value: publication.arxiv },
    ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <header className="detail-header section-shell">
        <a href="/" className="brand" aria-label="Back to Shengtao Zhang home">
          <span className="brand-mark" aria-hidden="true">S</span>
          <span>Shengtao Zhang</span>
        </a>
        <a href="/#publications">← All publications</a>
      </header>

      <main className="publication-detail section-shell">
        <div className="detail-hero">
          <p className="eyebrow">
            {publication.venue} · {publication.year}
          </p>
          <h1>{publication.title}</h1>
          <p className="detail-authors">
            {publication.authors.map((author, index) => (
              <span key={author}>
                {author === "Shengtao Zhang" ? <strong>{author}</strong> : author}
                {index < publication.authors.length - 1 ? ", " : ""}
              </span>
            ))}
          </p>
          <div className="detail-tags">
            {publication.tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
          <div className="hero-actions detail-actions">
            <a className="button-primary" href={publication.paperUrl} target="_blank" rel="noreferrer">
              Read paper ↗
            </a>
            <a href={publication.pdfUrl} target="_blank" rel="noreferrer">
              PDF ↗
            </a>
            {publication.codeUrl ? (
              <a href={publication.codeUrl} target="_blank" rel="noreferrer">
                Code ↗
              </a>
            ) : null}
            {publication.projectUrl ? (
              <a href={publication.projectUrl} target="_blank" rel="noreferrer">
                Project ↗
              </a>
            ) : null}
          </div>
        </div>

        <div className="detail-grid">
          <article className="detail-abstract">
            <p className="eyebrow">Abstract</p>
            <h2>{publication.shortTitle}</h2>
            <p>{publication.abstract}</p>
            <blockquote>{publication.contribution}</blockquote>
          </article>

          <aside className="detail-citation" aria-labelledby="citation-heading">
            <p className="eyebrow">Citation</p>
            <h2 id="citation-heading">BibTeX</h2>
            <pre>{publication.bibtex}</pre>
            <a
              href={publication.scholarUrl}
              target="_blank"
              rel="noreferrer"
            >
              Find citation on Google Scholar ↗
            </a>
          </aside>
        </div>

        <section className="detail-footer-note">
          <p className="eyebrow">Persistent research record</p>
          <p>
            DOI <a href={`https://doi.org/${publication.doi}`}>{publication.doi}</a>
            {" · "}arXiv <a href={`https://arxiv.org/abs/${publication.arxiv}`}>{publication.arxiv}</a>
          </p>
        </section>
      </main>
    </>
  );
}
