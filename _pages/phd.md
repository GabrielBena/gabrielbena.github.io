---
layout: page
permalink: /phd/
title: phd thesis
nav: true
nav_order: 4
description: "Physical Constraints and Functional Demands shape Modular Neuromorphic Intelligence"
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
---

My doctoral thesis, defended at Imperial College London (Neural Reckoning Group, supervised by Prof. Dan Goodman), explores **modularity and self-organisation in neural networks**.

The first half of the thesis investigates the *structure–function relationship* in neural networks: how structural modularity, resource constraints, and input statistics jointly shape functional specialisation, and how compositional learning can be grounded in physically embedded, energy-constrained substrates such as memristive neuromorphic hardware.

The second half steps a level deeper, into the foundations of *self-organisation* itself. It explores how continuous Neural Cellular Automata can be sculpted into a [universal computational medium]({{ '/blog/2025/bena2025unca/' | relative_url }}) via gradient descent, and how a closely related local-message-passing policy can grow and self-repair [digital Boolean circuits]({{ '/blog/2026/bena2026sodc/' | relative_url }}) — bridging biological resilience and reconfigurable hardware.

Together, these chapters frame modularity less as a fixed architectural property and more as an *emergent phenomenon* arising from the tension between constraints and goals, between local rules and global behaviour.

<div class="pdf-toolbar">
  <a class="btn btn-sm btn-outline-primary"
     href="{{ page.pdf | prepend: '/assets/pdf/' | relative_url }}"
     target="_blank" rel="noopener noreferrer">
    <i class="fa-solid fa-arrow-up-right-from-square"></i> Open in new tab
  </a>
  <a class="btn btn-sm btn-outline-secondary"
     href="{{ page.pdf | prepend: '/assets/pdf/' | relative_url }}"
     download>
    <i class="fa-solid fa-download"></i> Download PDF
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
