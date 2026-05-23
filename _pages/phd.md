---
layout: page
permalink: /phd/
title: "Physical Constraints and Functional Demands shape Modular Neuromorphic Intelligence"
nav_title: phd thesis
seo_title: "Physical Constraints and Functional Demands shape Modular Neuromorphic Intelligence — PhD thesis (Gabriel Béna, Imperial College London, 2026)"
nav: true
nav_order: 4
description: "PhD Thesis · Imperial College London · 2026"
keywords: "Gabriel Béna, PhD thesis, Imperial College London, Dan Goodman, Neural Reckoning, modularity, modular neural networks, neuromorphic computing, neural cellular automata, self-organisation, self-organising digital circuits, Boolean circuits, spiking neural networks, compositional learning, structure-function, memristive hardware"
og_image: /assets/img/phd_cover.png
pdf: phd_thesis.pdf
_styles: >
  .pdf-toolbar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0 0 1.25rem 0;
    flex-wrap: wrap;
  }
  .pdf-toolbar .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .pdf-embed-wrapper {
    position: relative;
    width: 100%;
    height: 90vh;
    min-height: 700px;
    border: 1px solid var(--global-divider-color, #ddd);
    border-radius: 8px;
    overflow: hidden;
    background: #f7f7f7;
  }
  .pdf-embed-wrapper embed,
  .pdf-embed-wrapper object,
  .pdf-embed-wrapper iframe {
    width: 100%;
    height: 100%;
    border: 0;
    display: block;
  }
  .pdf-fallback {
    padding: 1rem;
    font-size: 0.95rem;
    color: var(--global-text-color, #333);
  }
  @media (max-width: 768px) {
    .pdf-embed-wrapper { height: 70vh; min-height: 480px; }
  }
  .thesis-cite {
    margin-top: 2rem;
    padding: 1rem 1.25rem;
    border: 1px solid var(--global-divider-color, #ddd);
    border-radius: 8px;
    background: var(--global-bg-color, #fafafa);
  }
  .thesis-cite-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
  }
  .thesis-cite-header h3 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
  }
  .thesis-cite-meta {
    font-size: 0.95rem;
    margin: 0 0 0.75rem 0;
    color: var(--global-text-color, #333);
  }
  .thesis-cite pre {
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.45;
    max-height: 280px;
    overflow: auto;
  }
---

<p class="text-muted" style="font-size:0.95rem; margin-bottom:1rem;">
  <strong>Gabriel Béna</strong> ·
  <a href="https://doi.org/10.25560/128990" target="_blank" rel="noopener noreferrer">doi:10.25560/128990</a>
</p>

My doctoral thesis, defended at Imperial College London (Neural Reckoning Group, supervised by Prof. Dan Goodman), explores **modularity and self-organisation in neural networks**.

The first half of the thesis investigates the _structure–function relationship_ in neural networks: how structural modularity, resource constraints, and input statistics jointly shape functional specialisation, and how compositional learning can be grounded in physically embedded, energy-constrained substrates such as memristive neuromorphic hardware.

The second half steps a level deeper, into the foundations of _self-organisation_ itself. It explores how continuous Neural Cellular Automata can be sculpted into a [universal computational medium]({{ '/blog/2025/bena2025unca/' | relative_url }}) via gradient descent, and how a closely related local-message-passing policy can grow and self-repair [digital Boolean circuits]({{ '/blog/2026/bena2026sodc/' | relative_url }}) — bridging biological resilience and reconfigurable hardware.

Together, these chapters frame modularity less as a fixed architectural property and more as an _emergent phenomenon_ arising from the tension between constraints and goals, between local rules and global behaviour.

<div class="pdf-toolbar">
  <a class="btn btn-sm btn-outline-secondary"
     href="{{ page.pdf | prepend: '/assets/pdf/' | relative_url }}"
     download>
    <i class="fa-solid fa-download"></i> Download PDF
  </a>
  <a class="btn btn-sm btn-outline-info"
     href="https://doi.org/10.25560/128990"
     target="_blank" rel="noopener noreferrer"
     title="Official record on Imperial College London's Spiral repository">
    <i class="fa-solid fa-book"></i> View on Spiral (DOI)
  </a>
</div>

<div class="pdf-embed-wrapper">
  <object
    data="{{ page.pdf | prepend: '/assets/pdf/' | relative_url }}#view=FitH&toolbar=1&navpanes=0"
    type="application/pdf"
    aria-label="PhD Manuscript">
    <embed
      src="{{ page.pdf | prepend: '/assets/pdf/' | relative_url }}#view=FitH&toolbar=1&navpanes=0"
      type="application/pdf" />
    <div class="pdf-fallback">
      <p>Your browser can't display PDFs inline.
        <a href="{{ page.pdf | prepend: '/assets/pdf/' | relative_url }}" target="_blank" rel="noopener noreferrer">Open the manuscript in a new tab</a>
        or use the download button above.
      </p>
    </div>
  </object>
</div>

<div class="thesis-cite">
  <div class="thesis-cite-header">
    <h3>Cite this thesis</h3>
    <a class="btn btn-sm btn-outline-secondary"
       href="{{ '/assets/bibliography/phd_cite.bib' | relative_url }}"
       download>
      <i class="fa-solid fa-download"></i> .bib
    </a>
  </div>
  <p class="thesis-cite-meta">
    Béna, G. (2026). <em>Physical constraints and functional demands shape modular neuromorphic intelligence</em> [PhD thesis, Imperial College London]. <a href="https://doi.org/10.25560/128990" target="_blank" rel="noopener noreferrer">https://doi.org/10.25560/128990</a>
  </p>

{% highlight bibtex %}
@phdthesis{bena2026modular,
title = {Physical constraints and functional demands shape modular neuromorphic intelligence},
author = {B{\'e}na, Gabriel},
school = {Imperial College London},
year = {2026},
month = mar,
doi = {10.25560/128990},
url = {https://hdl.handle.net/10044/1/128990}
}
{% endhighlight %}

</div>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Thesis",
  "name": "Physical constraints and functional demands shape modular neuromorphic intelligence",
  "headline": "Physical constraints and functional demands shape modular neuromorphic intelligence",
  "alternateName": "Physical Constraints and Functional Demands shape Modular Neuromorphic Intelligence",
  "inSupportOf": "Doctor of Philosophy (PhD)",
  "author": {
    "@type": "Person",
    "name": "Gabriel Béna",
    "givenName": "Gabriel",
    "familyName": "Béna",
    "affiliation": {
      "@type": "CollegeOrUniversity",
      "name": "Imperial College London"
    },
    "sameAs": [
      {% if site.orcid_id %}"https://orcid.org/{{ site.orcid_id }}"{% if site.data.socials.scholar_userid %},{% endif %}{% endif %}
      {% if site.data.socials.scholar_userid %}"https://scholar.google.com/citations?user={{ site.data.socials.scholar_userid }}"{% endif %}
    ]
  },
  "datePublished": "2026-03",
  "inLanguage": "en",
  "publisher": {
    "@type": "CollegeOrUniversity",
    "name": "Imperial College London"
  },
  "url": "{{ page.url | prepend: site.baseurl | prepend: site.url }}",
  "sameAs": [
    "https://doi.org/10.25560/128990",
    "https://hdl.handle.net/10044/1/128990"
  ],
  "identifier": [
    { "@type": "PropertyValue", "propertyID": "DOI", "value": "10.25560/128990" },
    { "@type": "PropertyValue", "propertyID": "HDL", "value": "10044/1/128990" }
  ],
  "image": "{{ '/assets/img/phd_cover.png' | prepend: site.baseurl | prepend: site.url }}",
  "keywords": "modularity, self-organisation, neural cellular automata, neuromorphic computing, spiking neural networks, Boolean circuits, self-repairing circuits, compositional learning, structure-function, memristive hardware",
  "about": [
    "Modularity in neural networks",
    "Self-organisation",
    "Neural Cellular Automata",
    "Neuromorphic computing",
    "Self-organising digital circuits"
  ],
  "description": "Doctoral thesis investigating modularity and self-organisation in neural networks: how physical constraints (resource budgets, spatial embedding, wiring cost) and functional demands jointly shape modular specialisation, and how local message-passing rules can grow and self-repair Neural Cellular Automata and Boolean circuits."
}
</script>
