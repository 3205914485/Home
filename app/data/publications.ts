export type Publication = {
  slug: string;
  title: string;
  shortTitle: string;
  authors: string[];
  venue: string;
  year: number;
  tags: string[];
  paperUrl: string;
  pdfUrl: string;
  codeUrl?: string;
  projectUrl?: string;
  doi: string;
  arxiv: string;
  scholarUrl: string;
  scholarCitations: number;
  featured?: boolean;
  featuredFigure?: {
    src: string;
    width: number;
    height: number;
    alt: string;
    label: string;
    caption: string;
    sourceUrl: string;
  };
  contribution: string;
  abstract: string;
  bibtex: string;
};

export const publications: Publication[] = [
  {
    slug: "memrl",
    title: "MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory",
    shortTitle: "MemRL",
    authors: [
      "Shengtao Zhang",
      "Jiaqian Wang",
      "Ruiwen Zhou",
      "Junwei Liao",
      "Yuchen Feng",
      "Zhuo Li",
      "Yujie Zheng",
      "Weinan Zhang",
      "Ying Wen",
      "Zhiyu Li",
      "Feiyu Xiong",
      "Yutao Qi",
      "Bo Tang",
      "Muning Wen",
    ],
    venue: "arXiv preprint",
    year: 2026,
    tags: ["Agents", "RL", "Memory"],
    paperUrl: "https://arxiv.org/abs/2601.03192",
    pdfUrl: "https://arxiv.org/pdf/2601.03192",
    codeUrl: "https://github.com/MemTensor/MemRL",
    doi: "10.48550/arXiv.2601.03192",
    arxiv: "2601.03192",
    scholarUrl:
      "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9nwcqAIAAAAJ&citation_for_view=9nwcqAIAAAAJ:9yKSN-GCB0IC",
    scholarCitations: 76,
    featured: true,
    featuredFigure: {
      src: "/papers/memrl-overview.png",
      width: 2367,
      height: 1331,
      alt: "MemRL concept figure contrasting stable human reasoning with a frozen language model and an evolving value-aware episodic memory.",
      label: "Runtime learning",
      caption:
        "A frozen LLM preserves stable reasoning. Its evolving episodic memory stores intent, attempts, and utility, then learns better value-aware retrieval from environmental feedback.",
      sourceUrl: "https://arxiv.org/abs/2601.03192",
    },
    contribution:
      "A non-parametric route to runtime self-improvement: keep the reasoner stable, let episodic memory learn utility from feedback.",
    abstract:
      "MemRL studies how a frozen language-model agent can improve during deployment without changing its weights. It separates stable reasoning from an evolving episodic memory, retrieves candidates by semantic relevance, and then selects useful experiences through learned Q-values. Environmental feedback continually refines those values, enabling the agent to retain high-utility strategies and suppress similar but unhelpful memories across reasoning, coding, embodied, and lifelong-agent benchmarks.",
    bibtex: `@article{zhang2026memrl,
  title={MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory},
  author={Zhang, Shengtao and Wang, Jiaqian and Zhou, Ruiwen and Liao, Junwei and Feng, Yuchen and Li, Zhuo and Zheng, Yujie and Zhang, Weinan and Wen, Ying and Li, Zhiyu and Xiong, Feiyu and Qi, Yutao and Tang, Bo and Wen, Muning},
  journal={arXiv preprint arXiv:2601.03192},
  year={2026}
}`,
  },
  {
    slug: "memq",
    title: "MemQ: Integrating Q-Learning into Self-Evolving Memory Agents over Provenance DAGs",
    shortTitle: "MemQ",
    authors: [
      "Junwei Liao",
      "Haoting Shi",
      "Ruiwen Zhou",
      "Jiaqian Wang",
      "Shengtao Zhang",
      "Wei Zhang",
      "Ying Wen",
      "Zhiyu Li",
      "Feiyu Xiong",
      "Bo Tang",
      "Weinan Zhang",
      "Muning Wen",
    ],
    venue: "arXiv preprint",
    year: 2026,
    tags: ["Agents", "RL", "Memory"],
    paperUrl: "https://arxiv.org/abs/2605.08374",
    pdfUrl: "https://arxiv.org/pdf/2605.08374",
    codeUrl: "https://github.com/jwliao-ai/MemQ",
    doi: "10.48550/arXiv.2605.08374",
    arxiv: "2605.08374",
    scholarUrl:
      "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9nwcqAIAAAAJ&citation_for_view=9nwcqAIAAAAJ:UeHWp8X0CEIC",
    scholarCitations: 0,
    featured: true,
    featuredFigure: {
      src: "/papers/memq-overview.png",
      width: 2317,
      height: 1094,
      alt: "MemQ lifecycle from a provenance DAG and Q-guided retrieval through environment feedback to backward credit propagation.",
      label: "Memory lifecycle",
      caption:
        "Each interaction retrieves memories, creates a new episode, and records its provenance. TD feedback then travels backward through the DAG to update the memories that enabled success.",
      sourceUrl: "https://arxiv.org/abs/2605.08374",
    },
    contribution:
      "Credit assignment for agent memory, propagated through the provenance chains that make later memories possible.",
    abstract:
      "MemQ models memory evolution as a structured credit-assignment problem. A provenance DAG records which retrieved memories contributed to each newly created memory, while TD-style eligibility traces propagate feedback backward through those dependencies. This makes retrieval utility reflect both immediate usefulness and downstream influence, improving self-evolving memory agents across multi-step interaction, code, embodied reasoning, and expert QA tasks.",
    bibtex: `@article{liao2026memq,
  title={MemQ: Integrating Q-Learning into Self-Evolving Memory Agents over Provenance DAGs},
  author={Liao, Junwei and Shi, Haoting and Zhou, Ruiwen and Wang, Jiaqian and Zhang, Shengtao and Zhang, Wei and Wen, Ying and Li, Zhiyu and Xiong, Feiyu and Tang, Bo and Zhang, Weinan and Wen, Muning},
  journal={arXiv preprint arXiv:2605.08374},
  year={2026}
}`,
  },
  {
    slug: "evokernel",
    title: "Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis",
    shortTitle: "EvoKernel",
    authors: [
      "Yujie Zheng",
      "Zhuo Li",
      "Shengtao Zhang",
      "Jiaqian Wang",
      "Junjie Sheng",
      "Junchi Yan",
      "Weinan Zhang",
      "Ying Wen",
      "Bo Tang",
      "Muning Wen",
    ],
    venue: "ICML",
    year: 2026,
    tags: ["Agents", "RL", "Memory", "Systems"],
    paperUrl: "https://openreview.net/forum?id=ajHTru25Kd",
    pdfUrl:
      "https://openreview.net/pdf/22928270cbb3cec9d649ad5a1a9275e1c4403016.pdf",
    projectUrl: "https://evokernel.zhuo.li",
    doi: "10.48550/arXiv.2603.10846",
    arxiv: "2603.10846",
    scholarUrl:
      "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9nwcqAIAAAAJ&citation_for_view=9nwcqAIAAAAJ:2osOgNQ5qMEC",
    scholarCitations: 1,
    featured: true,
    featuredFigure: {
      src: "/papers/evokernel-overview.png",
      width: 1985,
      height: 735,
      alt: "EvoKernel framework connecting cold-start drafting, value-driven memory and verification, and continual kernel refinement.",
      label: "Draft-to-refine loop",
      caption:
        "Cold-start drafting retrieves transferable experience for an initial kernel. Verification rewards update the shared memory, which later reuses successful traces for continual latency refinement.",
      sourceUrl: "https://openreview.net/forum?id=ajHTru25Kd",
    },
    contribution:
      "Value-guided experience reuse turns sparse NPU feedback into a continual drafting-and-refinement loop.",
    abstract:
      "Published at ICML 2026, EvoKernel addresses cold-start code generation in data-scarce accelerator ecosystems. It casts NPU kernel synthesis as a memory-based reinforcement-learning process, learns stage-specific experience values for initial drafting and later latency refinement, and shares useful experience across tasks. The resulting agent accumulates practical optimization knowledge online instead of relying on expensive domain-specific fine-tuning.",
    bibtex: `@inproceedings{zheng2026evokernel,
  title={Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis},
  author={Zheng, Yujie and Li, Zhuo and Zhang, Shengtao and Wang, Jiaqian and Sheng, Junjie and Yan, Junchi and Zhang, Weinan and Wen, Ying and Tang, Bo and Wen, Muning},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={306},
  year={2026}
}`,
  },
  {
    slug: "bus-cot",
    title: "A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering All Histopathology Categories",
    shortTitle: "BUS-CoT",
    authors: [
      "Haojun Yu",
      "Youcheng Li",
      "Zihan Niu",
      "Nan Zhang",
      "Xuantong Gong",
      "Huan Li",
      "Zhiying Zou",
      "Haifeng Qi",
      "Zhenxiao Cao",
      "Zijie Lan",
      "Xingjian Yuan",
      "Jiating He",
      "Haokai Zhang",
      "Shengtao Zhang",
      "Zicheng Wang",
      "Dong Wang",
      "Ziwei Zhao",
      "Congying Chen",
      "Yong Wang",
      "Wangyan Qin",
      "Qingli Zhu",
      "Liwei Wang",
    ],
    venue: "Scientific Data",
    year: 2026,
    tags: ["Multimodal", "Data", "Reasoning"],
    paperUrl: "https://www.nature.com/articles/s41597-026-06702-9",
    pdfUrl: "https://www.nature.com/articles/s41597-026-06702-9.pdf",
    codeUrl: "https://doi.org/10.5281/zenodo.17870860",
    projectUrl: "https://doi.org/10.6084/m9.figshare.30838715",
    doi: "10.1038/s41597-026-06702-9",
    arxiv: "2509.17046",
    scholarUrl:
      "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9nwcqAIAAAAJ&citation_for_view=9nwcqAIAAAAJ:d1gkVwhDpl0C",
    scholarCitations: 3,
    contribution:
      "A reasoning-rich breast ultrasound resource that connects images, pathology coverage, and clinically grounded chains of thought.",
    abstract:
      "BUS-CoT is a multimodal breast-ultrasound dataset designed to cover the full range of histopathology categories while pairing visual evidence with structured diagnostic reasoning. It supports research on clinically grounded chain-of-thought, robust multimodal understanding, and evaluation across diverse lesion types, with public data and reproducible resources released alongside the paper.",
    bibtex: `@article{yu2026buscot,
  title={A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering All Histopathology Categories},
  author={Yu, Haojun and Li, Youcheng and Niu, Zihan and Zhang, Nan and Gong, Xuantong and Li, Huan and Zou, Zhiying and Qi, Haifeng and Cao, Zhenxiao and Lan, Zijie and Yuan, Xingjian and He, Jiating and Zhang, Haokai and Zhang, Shengtao and Wang, Zicheng and Wang, Dong and Zhao, Ziwei and Chen, Congying and Wang, Yong and Qin, Wangyan and Zhu, Qingli and Wang, Liwei},
  journal={Scientific Data},
  volume={13},
  pages={370},
  doi={10.1038/s41597-026-06702-9},
  year={2026}
}`,
  },
  {
    slug: "ptcl",
    title: "PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph",
    shortTitle: "PTCL",
    authors: [
      "Shengtao Zhang",
      "Haokai Zhang",
      "Shiqi Lou",
      "Zicheng Wang",
      "Zinan Zeng",
      "Yilin Wang",
      "Minnan Luo",
    ],
    venue: "arXiv preprint",
    year: 2025,
    tags: ["Graph Learning", "Curriculum Learning"],
    paperUrl: "https://arxiv.org/abs/2504.17641",
    pdfUrl: "https://arxiv.org/pdf/2504.17641",
    codeUrl: "https://github.com/3205914485/FLiD",
    doi: "10.48550/arXiv.2504.17641",
    arxiv: "2504.17641",
    scholarUrl:
      "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9nwcqAIAAAAJ&citation_for_view=9nwcqAIAAAAJ:u5HHmVD_uO8C",
    scholarCitations: 0,
    contribution:
      "Temporal curriculum learning makes final-timestamp labels useful across the full evolution of a dynamic graph.",
    abstract:
      "PTCL targets dynamic node classification when only final-timestamp labels are available. A decoupled architecture learns time-aware representations while producing pseudo-labels aligned with the observed final labels, and a temporal curriculum weights pseudo-labels according to their reliability over time. The work also introduces CoOAG and the FLiD evaluation framework for label-limited dynamic graphs.",
    bibtex: `@article{zhang2025ptcl,
  title={PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph},
  author={Zhang, Shengtao and Zhang, Haokai and Lou, Shiqi and Wang, Zicheng and Zeng, Zinan and Wang, Yilin and Luo, Minnan},
  journal={arXiv preprint arXiv:2504.17641},
  year={2025}
}`,
  },
  {
    slug: "unveiling-the-hidden",
    title: "Unveiling the Hidden: Movie Genre and User Bias in Spoiler Detection",
    shortTitle: "Unveiling the Hidden",
    authors: [
      "Haokai Zhang",
      "Shengtao Zhang",
      "Zijian Cai",
      "Heng Wang",
      "Ruixuan Zhu",
      "Zinan Zeng",
      "Minnan Luo",
    ],
    venue: "ECML PKDD",
    year: 2025,
    tags: ["Graph Learning", "Language"],
    paperUrl: "https://doi.org/10.1007/978-3-032-06066-2_2",
    pdfUrl: "https://arxiv.org/pdf/2504.17834",
    codeUrl: "https://github.com/AI-explorer-123/GUSD",
    doi: "10.1007/978-3-032-06066-2_2",
    arxiv: "2504.17834",
    scholarUrl:
      "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9nwcqAIAAAAJ&citation_for_view=9nwcqAIAAAAJ:u-x6o8ySG0sC",
    scholarCitations: 1,
    contribution:
      "Genre structure and user history expose spoiler patterns that text-only classifiers tend to miss.",
    abstract:
      "This work studies how movie genre and user-specific behavior shape spoiler likelihood. It combines dynamically modeled review history with genre-aware aggregation and a mixture-of-experts architecture, allowing the detector to capture both recurring author bias and genre-dependent language patterns. The approach improves spoiler detection across benchmark review datasets and was published at ECML PKDD 2025.",
    bibtex: `@inproceedings{zhang2025unveiling,
  title={Unveiling the Hidden: Movie Genre and User Bias in Spoiler Detection},
  author={Zhang, Haokai and Zhang, Shengtao and Cai, Zijian and Wang, Heng and Zhu, Ruixuan and Zeng, Zinan and Luo, Minnan},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
  pages={19--36},
  year={2025},
  doi={10.1007/978-3-032-06066-2_2}
}`,
  },
];

export const publicationBySlug = new Map(
  publications.map((publication) => [publication.slug, publication]),
);

export const scholarSearchUrl =
  "https://scholar.google.com/citations?user=9nwcqAIAAAAJ&hl=en";

export const scholarCitationsCheckedAt = "2026-07-13";

export function scholarCiteUrl(title: string) {
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(`allintitle: ${title}`)}`;
}
