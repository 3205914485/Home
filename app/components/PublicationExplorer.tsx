"use client";

import { useMemo, useRef, useState } from "react";
import {
  publications as canonicalPublications,
  scholarCitationsCheckedAt,
  type Publication,
} from "../data/publications";

const topics = [
  "All",
  "Agents",
  "RL",
  "Memory",
  "Graph Learning",
  "Language",
  "Multimodal",
];

const normalize = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

function Authors({ names }: { names: string[] }) {
  return (
    <p className="publication-authors">
      {names.map((name, index) => (
        <span key={name}>
          {name === "Shengtao Zhang" ? <strong>{name}</strong> : name}
          {index < names.length - 1 ? ", " : ""}
        </span>
      ))}
    </p>
  );
}

export function PublicationExplorer() {
  const [query, setQuery] = useState("");
  const [topic, setTopic] = useState("All");
  const [activeCitation, setActiveCitation] = useState<Publication | null>(null);
  const [copyState, setCopyState] = useState("Copy BibTeX");
  const dialogRef = useRef<HTMLDialogElement>(null);

  const filtered = useMemo(() => {
    const normalizedQuery = normalize(query);
    return canonicalPublications.filter((publication) => {
      const matchesTopic =
        topic === "All" || publication.tags.includes(topic);
      const haystack = normalize(
        `${publication.title} ${publication.authors.join(" ")} ${publication.venue} ${publication.tags.join(" ")}`,
      );
      return matchesTopic && (!normalizedQuery || haystack.includes(normalizedQuery));
    });
  }, [query, topic]);

  function openCitation(publication: Publication) {
    setActiveCitation(publication);
    setCopyState("Copy BibTeX");
    dialogRef.current?.showModal();
  }

  async function copyBibtex() {
    if (!activeCitation) return;
    await navigator.clipboard.writeText(activeCitation.bibtex);
    setCopyState("Copied");
  }

  return (
    <>
      <div className="publication-tools">
        <label className="search-field">
          <span className="sr-only">Search publications</span>
          <span aria-hidden="true">⌕</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search title, venue, or topic"
          />
        </label>
      </div>

      <div className="topic-filter" aria-label="Filter publications by topic">
        {topics.map((value) => (
          <button
            key={value}
            type="button"
            className={topic === value ? "active" : ""}
            aria-pressed={topic === value}
            onClick={() => setTopic(value)}
          >
            {value}
          </button>
        ))}
      </div>

      <div className="publication-list" aria-live="polite">
        {filtered.map((publication, index) => (
          <article className="publication-row" key={publication.slug}>
            <div className="publication-number" aria-hidden="true">
              {String(index + 1).padStart(2, "0")}
            </div>
            <div className="publication-content">
              <div className="publication-kicker">
                <span>{publication.venue}</span>
                <span>{publication.year}</span>
                <a
                  className="micro-tag citation-count"
                  href={publication.scholarUrl}
                  target="_blank"
                  rel="noreferrer"
                  title={`Google Scholar citations, checked ${scholarCitationsCheckedAt}`}
                >
                  Cited by {publication.scholarCitations}
                </a>
                {publication.tags.slice(0, 3).map((tag) => (
                  <span className="micro-tag" key={tag}>
                    {tag}
                  </span>
                ))}
              </div>
              <h3>
                <a href={`/publications/${publication.slug}`}>
                  {publication.title}
                </a>
              </h3>
              <Authors names={publication.authors} />
              <p className="publication-contribution">
                {publication.contribution}
              </p>
              <div className="publication-actions">
                <a href={`/publications/${publication.slug}`}>Details</a>
                <a href={publication.paperUrl} target="_blank" rel="noreferrer">
                  Paper ↗
                </a>
                {publication.codeUrl ? (
                  <a href={publication.codeUrl} target="_blank" rel="noreferrer">
                    Code ↗
                  </a>
                ) : null}
                {publication.projectUrl ? (
                  <a
                    href={publication.projectUrl}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Project ↗
                  </a>
                ) : null}
                <button type="button" onClick={() => openCitation(publication)}>
                  Cite
                </button>
              </div>
            </div>
          </article>
        ))}
        {filtered.length === 0 ? (
          <div className="empty-state">
            <p>No publications match this filter.</p>
            <button
              type="button"
              className="text-button"
              onClick={() => {
                setQuery("");
                setTopic("All");
              }}
            >
              Clear filters
            </button>
          </div>
        ) : null}
      </div>

      <dialog
        ref={dialogRef}
        className="cite-dialog"
        onClose={() => setActiveCitation(null)}
      >
        {activeCitation ? (
          <div>
            <div className="dialog-heading">
              <div>
                <p className="eyebrow">Cite this work</p>
                <h2>{activeCitation.shortTitle}</h2>
              </div>
              <button
                type="button"
                className="dialog-close"
                aria-label="Close citation dialog"
                onClick={() => dialogRef.current?.close()}
              >
                ×
              </button>
            </div>
            <pre>{activeCitation.bibtex}</pre>
            <div className="dialog-actions">
              <button type="button" className="button-primary" onClick={copyBibtex}>
                {copyState}
              </button>
              <a
                href={activeCitation.scholarUrl}
                target="_blank"
                rel="noreferrer"
              >
                Cited by {activeCitation.scholarCitations} on Google Scholar ↗
              </a>
            </div>
            <span className="sr-only" aria-live="polite">
              {copyState === "Copied" ? "BibTeX copied to clipboard" : ""}
            </span>
          </div>
        ) : null}
      </dialog>
    </>
  );
}
