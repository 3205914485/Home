import { PublicationExplorer } from "./components/PublicationExplorer";
import { publications, scholarSearchUrl } from "./data/publications";
import { getSiteOrigin } from "./lib/site-url";

const featured = publications.filter((publication) => publication.featured);

const researchAreas = [
  {
    index: "01",
    title: "Agent systems",
    label: "Act",
    copy: "Agents that plan, use tools, and improve through interaction instead of staying fixed after deployment.",
  },
  {
    index: "02",
    title: "Runtime reinforcement learning",
    label: "Adapt",
    copy: "Feedback-driven learning loops that turn execution outcomes into better choices at inference time.",
  },
  {
    index: "03",
    title: "Memory",
    label: "Remember",
    copy: "Episodic and value-aware memory that retrieves experience for utility, provenance, and continual refinement.",
  },
];

const news = [
  {
    date: "2026.05",
    title: "MemQ released",
    copy: "Credit assignment for self-evolving memory agents over provenance DAGs.",
    href: "https://arxiv.org/abs/2605.08374",
  },
  {
    date: "2026.03",
    title: "EvoKernel released",
    copy: "Value-driven memory for cold-start NPU kernel synthesis and continual refinement.",
    href: "https://arxiv.org/abs/2603.10846",
  },
  {
    date: "2026.01",
    title: "MemRL released",
    copy: "Runtime reinforcement learning on episodic memory for self-evolving agents.",
    href: "https://arxiv.org/abs/2601.03192",
  },
  {
    date: "2025.09",
    title: "Unveiling the Hidden at ECML PKDD",
    copy: "Genre- and user-aware modeling for spoiler detection.",
    href: "https://doi.org/10.1007/978-3-032-06066-2_2",
  },
];

export default async function Home() {
  const siteOrigin = await getSiteOrigin();
  const personJsonLd = {
    "@context": "https://schema.org",
    "@type": "Person",
    name: "Shengtao Zhang",
    url: siteOrigin,
    image: `${siteOrigin}/profile.jpg`,
    affiliation: {
      "@type": "Organization",
      name: "SJTU-MARL",
      url: "https://github.com/sjtu-marl",
    },
    alumniOf: {
      "@type": "CollegeOrUniversity",
      name: "Xi'an Jiaotong University",
    },
    knowsAbout: [
      "AI agents",
      "reinforcement learning",
      "agent memory",
      "runtime learning",
    ],
    sameAs: [
      "https://github.com/3205914485",
      "https://dblp.org/pid/389/1973",
      scholarSearchUrl,
    ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(personJsonLd) }}
      />

      <header className="site-header">
        <a className="brand" href="#top" aria-label="Shengtao Zhang, home">
          <span className="brand-mark" aria-hidden="true">S</span>
          <span>Shengtao Zhang</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#research">Research</a>
          <a href="#publications">Publications</a>
          <a href="#about">About</a>
        </nav>
        <a
          className="header-contact"
          href="mailto:zhangst@stu.xjtu.edu.cn"
        >
          Contact <span aria-hidden="true">↗</span>
        </a>
      </header>

      <main id="top">
        <section className="hero section-shell" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="eyebrow hero-eyebrow">
              <span className="status-pulse" aria-hidden="true" />
              Research at SJTU-MARL · Shanghai
            </p>
            <h1 id="hero-title">
              Agents that <em>remember,</em>
              <br />
              adapt, and improve
              <br />
              at runtime.
            </h1>
            <p className="hero-intro">
              I am <strong>Shengtao Zhang</strong>. I study self-evolving agents
              at the intersection of reinforcement learning and memory—systems
              that turn experience into better decisions without retraining the
              underlying model.
            </p>
            <div className="hero-actions">
              <a className="button-primary" href="#publications">
                Explore publications <span aria-hidden="true">↓</span>
              </a>
              <a
                href={scholarSearchUrl}
                target="_blank"
                rel="noreferrer"
                aria-label="Search Shengtao Zhang on Google Scholar"
              >
                Google Scholar ↗
              </a>
              <a
                href="https://github.com/3205914485"
                target="_blank"
                rel="noreferrer"
              >
                GitHub ↗
              </a>
            </div>
          </div>

          <div className="hero-visual" aria-label="Portrait of Shengtao Zhang">
            <div className="portrait-frame">
              <img src="/profile.jpg" alt="Shengtao Zhang" />
              <div className="portrait-caption">
                <span>SJTU-MARL</span>
                <span>Agent · RL · Memory</span>
              </div>
            </div>
            <div className="memory-map" aria-hidden="true">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
              <div className="memory-node node-memory">
                <span />
                Memory
              </div>
              <div className="memory-node node-policy">
                <span />
                Policy
              </div>
              <div className="memory-node node-feedback">
                <span />
                Feedback
              </div>
            </div>
          </div>

          <div className="hero-footnote">
            <span>Current focus</span>
            <p>
              How can an agent learn from every interaction while keeping its
              reasoning stable?
            </p>
          </div>
        </section>

        <section
          className="research-section section-shell"
          id="research"
          aria-labelledby="research-title"
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow">Research thesis</p>
              <h2 id="research-title">From storage to experience.</h2>
            </div>
            <p>
              My work connects three layers of adaptive intelligence: acting in
              an environment, learning from feedback, and preserving useful
              experience.
            </p>
          </div>
          <div className="research-grid">
            {researchAreas.map((area) => (
              <article className="research-card" key={area.index}>
                <div className="research-card-top">
                  <span>{area.index}</span>
                  <span>{area.label}</span>
                </div>
                <h3>{area.title}</h3>
                <p>{area.copy}</p>
                <div className="research-signal" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                  <i />
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="featured-section section-shell" aria-labelledby="featured-title">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Featured work</p>
              <h2 id="featured-title">Learning at inference time.</h2>
            </div>
            <a href="#publications">All publications ↓</a>
          </div>
          <div className="featured-grid">
            {featured.map((publication, index) => (
              <article
                className={`featured-card featured-card-${index + 1}`}
                key={publication.slug}
              >
                <div className="featured-meta">
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <span>{publication.year}</span>
                </div>
                <div>
                  <p className="featured-tags">{publication.tags.join(" · ")}</p>
                  <h3>
                    <a href={`/publications/${publication.slug}`}>
                      {publication.shortTitle}
                    </a>
                  </h3>
                  <p>{publication.contribution}</p>
                </div>
                <div className="featured-links">
                  <a href={`/publications/${publication.slug}`}>Read overview</a>
                  <a href={publication.paperUrl} target="_blank" rel="noreferrer">
                    Paper ↗
                  </a>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section
          className="publications-section section-shell"
          id="publications"
          aria-labelledby="publications-title"
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow">Selected publications</p>
              <h2 id="publications-title">Research index.</h2>
            </div>
            <p>
              A verified, topic-filterable index. Citation signals refresh from
              OpenAlex while the publication list remains manually reviewed
              against Google Scholar.
            </p>
          </div>
          <PublicationExplorer />
        </section>

        <section className="news-section section-shell" aria-labelledby="news-title">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Recent notes</p>
              <h2 id="news-title">News.</h2>
            </div>
          </div>
          <div className="news-list">
            {news.map((item) => (
              <a href={item.href} target="_blank" rel="noreferrer" key={item.date}>
                <time>{item.date}</time>
                <span className="news-title">{item.title}</span>
                <span>{item.copy}</span>
                <b aria-hidden="true">↗</b>
              </a>
            ))}
          </div>
        </section>

        <section
          className="about-section section-shell"
          id="about"
          aria-labelledby="about-title"
        >
          <div className="about-main">
            <p className="eyebrow">About</p>
            <h2 id="about-title">
              Building learning loops for agents that operate in the real
              world.
            </h2>
            <p>
              I work on agent learning and memory with SJTU-MARL. I am
              especially interested in non-parametric adaptation: how agents
              can use outcomes, provenance, and episodic experience to improve
              continuously while the base model stays stable.
            </p>
            <p>
              Previously, I studied Artificial Intelligence at Xi&apos;an
              Jiaotong University and worked on dynamic graph learning,
              language understanding, and reliable decision systems.
            </p>
          </div>
          <aside className="about-aside" aria-label="Profile details">
            <div>
              <span>Affiliation</span>
              <a href="https://github.com/sjtu-marl" target="_blank" rel="noreferrer">
                SJTU-MARL ↗
              </a>
            </div>
            <div>
              <span>Research</span>
              <p>Agents · RL · Memory</p>
            </div>
            <div>
              <span>Education</span>
              <p>B.Eng. in Artificial Intelligence, Xi&apos;an Jiaotong University</p>
            </div>
            <div>
              <span>Profiles</span>
              <p>
                <a href="https://dblp.org/pid/389/1973" target="_blank" rel="noreferrer">
                  DBLP ↗
                </a>{" "}
                ·{" "}
                <a href={scholarSearchUrl} target="_blank" rel="noreferrer">
                  Scholar ↗
                </a>
              </p>
            </div>
          </aside>
        </section>
      </main>

      <footer className="site-footer section-shell">
        <div>
          <p className="footer-name">Shengtao Zhang</p>
          <p>Self-evolving agents through reinforcement learning and memory.</p>
        </div>
        <div className="footer-links">
          <a href={scholarSearchUrl} target="_blank" rel="noreferrer">
            Scholar ↗
          </a>
          <a href="https://dblp.org/pid/389/1973" target="_blank" rel="noreferrer">
            DBLP ↗
          </a>
          <a href="https://github.com/3205914485" target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
          <a href="mailto:zhangst@stu.xjtu.edu.cn">Email ↗</a>
        </div>
        <p className="footer-meta">© 2026 · Shanghai, China</p>
      </footer>
    </>
  );
}
