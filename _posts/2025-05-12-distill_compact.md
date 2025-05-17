---
layout: distill
title: A Path to Universal Neural Cellular Automata
description: Exploring how neural cellular automata can develop continuous universal computation through training by gradient descent
tags: neural-cellular-automata universal-computation
giscus_comments: true
date: 2025-05-12
featured: false
related_publications: true

authors:
  - name: Gabriel Béna
    url: "https://gabrielbena.github.io/"
    affiliations:
      name: Imperial College London
  - name: Maxence Faldor
#     url: ""  # Please add URL if available
#     affiliations:
#       name: Imperial College London
  - name: Dan Goodman
#     url: ""  # Please add URL if available
#     affiliations:
#       name: Imperial College London
  - name: Antoine Cully
#     url: ""  # Please add URL if available
#     affiliations:
#       name: Imperial College London


# Essential for good BibTeX and citation display:
slug: "bena2025unca" # e.g., bena2025unca - for the @article{key,...}
journal: "Genetic and Evolutionary Computation Conference (GECCO '25 Companion)"
doi: "10.1145/3712255.373431" # e.g., 10.23915/distill.00000
# url: "https://your.article/full-url" # Link to the article itself

bibliography: unca.bib

toc:
  - name: Introduction and Background
  - name: Methods
    subsections:
      - name: General Setup
      - name: Neural Cellular Automata
      - name: Hardware (Immutable State)
      - name: Tasks
      - name: Training
  - name: Experiments and Results
    subsections:
      - name: Task Training
      - name: MNIST Classifier emulation
      - name: Future directions - task composition and neural compiler
  - name: Conclusion

---

<!-- <d-article> -->

Cellular automata have long been celebrated for their ability to generate complex behaviors from simple, local rules, with well-known discrete models like Conway's Game of Life proven capable of universal computation. Recent advancements have extended cellular automata into continuous domains, raising the question of whether these systems retain the capacity for universal computation. In parallel, neural cellular automata have emerged as a powerful paradigm where rules are learned via gradient descent rather than manually designed. This work explores the potential of neural cellular automata to develop a continuous Universal Cellular Automaton through training by gradient descent. We introduce a cellular automaton model, objective functions and training strategies to guide neural cellular automata toward universal computation in a continuous setting. Our experiments demonstrate the successful training of fundamental computational primitives — such as matrix multiplication and transposition — culminating in the emulation of a neural network solving the MNIST digit classification task directly within the cellular automata state. These results represent a foundational step toward realizing analog general-purpose computers, with implications for understanding universal computation in continuous dynamics and advancing the automated discovery of complex cellular automata behaviors via machine learning.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/grant_task_evolution.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Our framework performs matrix operations (translation, multiplication, rotation) through continuous cellular interactions evolving over time (from t=0 to t=60). The inset reveals the core mechanism: cells communicating only with neighbors, and interacting with a local fixed (but learned) heterogeneous substrate, are able to collectively solve mathematical tasks without explicit algorithms.
</div>

## Introduction and Background

Cellular Automata (CA) represent a fascinating class of computational models that have captivated researchers across disciplines for their ability to produce complex behaviors from simple rules <d-cite key="computational_beauty"></d-cite>. At their core, CA typically consist of a grid of cells, each in one of a finite number of states, which evolve over discrete time steps based on a fixed deterministic rule. This rule, applied uniformly to all cells, governs the state transition of each cell based solely on its current state and those of its neighbors. Despite their simplicity, CA have become a cornerstone in studying emergence and complexity <d-cite key="wolfram_cellular_1984"></d-cite>.

Several discrete CA models, including Conway's Game of Life <d-cite key="cgol,life_turing_2016"></d-cite>, Rule 110 of Elementary Cellular Automata <d-cite key="wolfram,cook"></d-cite>, Langton's Ant <d-cite key="langton"></d-cite>, and Wireworld <d-cite key="wireworld"></d-cite>, have been mathematically proven Turing-complete, underscoring their remarkable power and expressiveness. These proofs establish that despite their simple construction, such systems can perform any computation that a traditional computer can. Beyond these formal demonstrations, researchers have constructed fully functional Turing machines within these CA, albeit through arduous efforts requiring meticulous design and substantial time investment. To ease this burden, evolutionary algorithms have been employed to automate CA rule discovery <d-cite key="mitchell,sapin"></d-cite>, though these approaches still target discrete systems and specific behaviors rather than general-purpose computation.

In recent years, the development of continuous CA has bridged the gap between discrete models and the analog characteristics of the real world. Notable examples include Lenia <d-cite key="lenia"></d-cite> and SmoothLife <d-cite key="smoothlife"></d-cite>, which extend classic CA to simulate lifelike patterns with continuous dynamics. While evolutionary search <d-cite key="reinke_intrinsically_2020,leniabreeder"></d-cite> and gradient descent <d-cite key="hamon_discovering_2024"></d-cite> have been applied to optimize continuous CA patterns, a key open question persists: are these models capable of universal computation? While the answer is likely affirmative given their expressive potential, proving this remains elusive. 

The lack of discrete states and well-defined transitions makes it harder to encode symbolic information reliably — slight perturbations can lead to significant divergence over time, undermining the stability required for computation. Moreover, continuous models often exhibit smooth, fuzzy dynamics that make it challenging to design modular components like wires, gates, or memory elements with predictable behavior. What was already a laborious task in discrete CA becomes more difficult, if not practically impossible, in the continuous domain, highlighting a fundamental challenge in building efficient analog computers.

Neural Cellular Automata (NCA) represent a paradigm shift by replacing hand-crafted rules with neural networks trained via gradient descent <d-cite key="nca"></d-cite>. Unlike traditional CA with explicitly handcrafted rules, NCA leverage differentiable architectures where the update rule is parameterized by a neural network and optimized end-to-end. This makes it possible to _learn_ complex behaviors from data, bypassing the need for manual rule design. NCAs have demonstrated success in tasks ranging from morphogenesis <d-cite key="nca"></d-cite> and classification <d-cite key="self_classifying"></d-cite> to solving reasoning challenges <d-cite key="cax"></d-cite> and growing neural networks <d-cite key="hypernca"></d-cite>. Given the Turing completeness of classical CA, NCA offer an exciting opportunity to search for computational rules through optimization, turning rule discovery into a machine learning problem. This shift is significant: the traditionally arduous task of hand-crafting rule sets that give rise to desired behaviors is now offloaded to the learning algorithm itself.

Our work connects to analog computing and neuromorphic systems, which leverage biological principles like local computations <d-cite key="small_world,modular_brain,Bullmore2012TheEO"></d-cite> and co-located memory and processing to overcome the von Neumann bottleneck <d-cite key="bottlneck"></d-cite>. Neuromorphic systems implement these principles through distributed processing elements with local memory, using mixed-signal circuits that approximate neural dynamics while maintaining energy efficiency <d-cite key="neuromorphic_mead,neuromorphic_review,schuman2017surveyneuromorphiccomputingneural"></d-cite>. By exploring how systems with local interactions can implement universal computation, we aim to develop computing architectures that balance computational power with the efficiency characteristic of biological intelligence.

In this work, we explore the potential of the Neural Cellular Automata paradigm to develop a continuous Universal Cellular Automata <d-cite key="universal_ca"></d-cite>, with the ambitious goal of inducing a universal Turing machine <d-cite key="universal_turing_machine"></d-cite> to emerge through gradient descent training. This has implications beyond academic curiosity, touching on fundamental questions about continuous dynamic systems' computational potential and the possibility of creating universal analog computers.

Our approach introduces: (1) a novel framework disentangling "hardware" and "state" within NCA, where CA rules serve as the physics dictating state transitions while hardware provides an immutable and heterogeneous scaffold; (2) objective functions and training setups steering NCA toward universal computation in a continuous setting; (3) experiments demonstrating the training of essential computational primitives; and (4) a demonstration of emulating a neural network directly within the CA state, solving the MNIST digit classification task. These results mark a critical first step toward sculpting continuous CA into powerful, general-purpose computational systems through gradient descent.

## Methods

We leverage the CAX <d-cite key="cax"></d-cite> library for high-performance neural cellular automata implementations and run our experiments on a single L40 GPU. Here we present a condensed overview of our approach.

### General Setup

Our framework demonstrates neural cellular automata's capability to act as a general computational substrate. We introduce a novel architecture that partitions the NCA state space into two distinct components:

- **Mutable state**: The main computational workspace where inputs are transformed into outputs. This state changes through time during task execution, governed by NCA rules.

- **Immutable state**: A specialized hardware configuration that remains fixed during any single experiment. This state can be monolithic (uniform across the grid) or modular (composed of different specialized components).

This separation enables a two-level optimization: at the global level, we train a general-purpose NCA rule to support diverse operations, while at the task-specific level, we optimize hardware configurations. The system adapts its dynamics using the local available hardware, similar to how different components on a motherboard enable different functions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/architecture.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Schematic of our architecture, showing the distinction between mutable (computational) and immutable (hardware) states.
</div>
### Neural Cellular Automata

Our NCA models solve tasks directly within their mutable state through local cell interactions, parametrized by a neural network with two key components:

1. **Perceive Function**: Gathers neighborhood information through learnable convolution filters applied to the immediate vicinity, transforming input states into a higher-dimensional perception vector.

2. **Update Function**: Utilizes the perception vector and local hardware vector to update each cell's state. We use an attention-based update module that calculates state updates by conditioning perceived information on the local hardware.

The attention mechanism computes weights over multiple processing heads using the hardware input: $\alpha = \text{softmax}((I \cdot W_{\text{embed}}) / T)$. The perception vector is processed through parallel pathways, generating potential update vectors for each head. The final state update is computed as a weighted sum of these vectors: $\Delta S = \sum_{h=1}^{N} \alpha_h V_h$, with cells updated residually: $S_{t+1} = S_t + \Delta S$.

This mechanism allows the NCA to dynamically adapt based on the local hardware, enabling diverse computations with the same underlying update rule.

### Hardware (Immutable State)

We explored two approaches for designing specialized hardware configurations:

#### Monolithic Hardware

Our initial implementation used task-specific parameters with the same spatial dimension as the computational state. While successful and visually interpretable, this approach lacks generalizability across different spatial configurations and matrix sizes.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/monolithic_hw.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Monolithic hardware configurations for 3 different sub-tasks
</div>

#### Modular Hardware

To address these limitations, we developed a modular approach with three purpose-specific hardware components:

1. An **input embedding vector** marking cells that receive inputs
2. An **output embedding vector** marking cells that will serve as output
3. A **task embedding vector** enabling the NCA to recognize the required transformation

These components are assembled for each task instance, balancing the scale-free nature of NCAs with the need for specialized hardware. This approach enables zero-shot generalization to unseen task configurations and composite task chaining.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/modular_hw.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Modular hardware configurations for 3 different sub-tasks, and different matrix sizes and placement
</div>

### Tasks

We implemented a flexible framework of matrix-based operations to train robust and versatile NCA models:

**Matrix Operations:**
- **Identity Mapping**: Moving a matrix to a different location
- **Matrix Multiplication**: Computing C = A × B
- **Transposition/Rotation**: Computing B = A^T or 90° rotation
- **Element-wise Operations**: Computing C = A · B

We used diverse input distributions (uniform, Gaussian, spatially correlated, discrete-valued, and sparse) to prevent overfitting to specific patterns. With the modular hardware approach, we enabled flexible placement with dynamic positioning, variable matrix sizes, and multiple inputs/outputs.

### Training

Our training framework simultaneously optimizes a single NCA rule across diverse task instances. Each task instance is represented by an initial state, a target final state, and a mask indicating relevant evaluation regions.

During training, we sample batches containing various tasks. For each instance, the NCA evolves the initial state over discrete time steps. We compute loss (typically masked MSE) between the final and target states within relevant regions, then update parameters using gradient-based optimization.

Parameters for the shared NCA rule and shared hardware components are updated based on gradients across all tasks, while task-specific modules are updated using only their corresponding task gradients. This joint optimization encourages a versatile NCA capable of executing multiple computational functions by adapting to the presented hardware.

## Experiments and Results

### Task Training

#### Joint Training

Our Neural Cellular Automata successfully mastered various matrix operations simultaneously through a shared update rule architecture with task-specific hardware components. This demonstrates that a single NCA can develop general computational principles that apply across different matrix tasks.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/gifs/Translation.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Neural Cellular Automata performing matrix translation. The NCA learns to move the input matrix to a target location.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/gifs/Rotation.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Neural Cellular Automata performing matrix rotation. The NCA learns to rotate the input matrix by 90 degrees.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/gifs/Multiplication.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Neural Cellular Automata performing matrix multiplication. The NCA learns to compute the product of two input matrices.
</div>

By simultaneously learning diverse operations (multiplication, translation, transposition, rotation), the model demonstrates mastery of a complete algebra of matrix operations—the essential building blocks for more complex computation. This multi-task foundation enables more sophisticated applications, including our MNIST classifier emulation.

#### Downstream Tasks Fine-tuning

A key advantage of our architecture is that once the NCA is pre-trained, new tasks can be accommodated by fine-tuning only the hardware configurations while keeping the core update rules frozen. In our experiments, this approach increases training speed by approximately 2x compared to full model retraining.

### MNIST Classifier emulation

We demonstrated a practical application by using our NCA to emulate a neural network directly in its computational workspace. We emulated a single-layer MLP solving the MNIST digit classification task.

First, we pre-trained a simple linear network to classify MNIST digits with sufficient accuracy. Our NCA model was pre-trained on smaller 8×8 matrix multiplication tasks. Since MNIST classification requires larger matrices (784×10), we implemented block-matrix decomposition, fragmenting the classification into multiple smaller operations that fit within the NCA's constraints.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/mnist.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    NCAs as a neural network emulator. Inputs show a batch of flattened MNIST images, alongside the weight matrix of a pre-trained single-layer linear classifier. We decompose this matrix multiplication into sub-blocks, that can be directly emulated in parallel by the NCA. Results are fetched from NCA states and aggregated back into logits, of size batch x 10 (figure shows the first 32 outputs only for readability). We compute accuracy by taking the logits argmax per batch and comparing with labels.
</div>

The NCA processed each block operation in parallel, after which we aggregated the results to reconstruct the complete classification logits. While we observed some accuracy degradation due to error propagation (60% accuracy for the emulated classification compared to 84% for the original), this demonstrates that neural network emulation via NCA is feasible.

This has significant implications for analog and physical computing. If our NCA's update rules were implemented as physical state transitions, this would represent a pathway toward physical neural network emulation without reverting to binary-level operations.

### Future directions: task composition and neural compiler

The modular hardware approach enables the creation of out-of-distribution tasks through component composition. We can design novel computational scenarios that the NCA wasn't explicitly trained on, such as duplicating a central matrix into multiple corners.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/OOD.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Examples of an Out of Distribution Task: the NCA needs to distribute a matrix in all corners in a larger grid than the one seen during training.
</div>

This framework opens the path toward complex composite tasks created through sequential chaining of primitive operations. For example:
1. Start with an input matrix and distribute copies to corner positions
2. Replace the hardware configuration to redefine these targets as inputs, then perform matrix multiplication toward a third corner
3. Update the hardware again to rotate the resulting matrix and return it to the original position

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/compiler.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Illustrating composite computational tasks using modular hardware configurations. Top panel: Multi-Rotation Sequential rotations resolving in the identity function. Bottom panel: Distribute-Multiply-Rotate task, showing a three-step process where matrices are distributed, multiplied, and rotated to achieve the target state. Each step shows both the hardware configuration and corresponding computational states.
</div>

While such sequences may appear simple, they demonstrate a critical capability: the NCA can execute complex algorithmic workflows through sequential hardware reconfiguration. This composite task chaining highlights the importance of stability in achieving composable computations.

We propose a dual-timestep approach to neural compilation:
- At the neuronal timestep, the NCA's mutable state evolves according to update rules
- At the compiler timestep, hardware parameters are reconfigured to guide high-level procedural steps

This separation mirrors classical computer architecture but within a continuous, differentiable substrate. As this approach matures, it could enable direct compilation of algorithms into neural cellular automata, combining neural network flexibility with programmatic precision.

#### Graph-based hardware hypernetwork

Taking the neural compilation idea further, we're developing a more principled graph-based hardware generation framework that offers significant improvements in flexibility and scale-invariance. This approach leverages a higher-level task representation abstraction where computational operations are modeled as a graph—nodes represent input and output regions while edges encode specific transformations between them.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0" style="background-color: white; border-radius: 8px; padding: 10px; border: 1px solid #ddd;">
        {% include figure.liquid path="assets/unca/hardware_meta_network.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Graph-based task representations and GNN-based hypernetwork for hardware generation
</div>

At the core of this system is a Hardware Meta-Network consisting of two key components:

1. A **Graph Neural Network (GNN) encoder** processes the task graph structure. Nodes contain normalized spatial information about input/output regions, and edges represent operations (matrix multiplication, rotation, etc.) to be performed between regions. Through message-passing layers, the GNN distills this representation into a fixed-dimensional latent task vector capturing the computational requirements.

2. This latent representation then conditions a **coordinate-based hypernetwork** that generates hardware vectors for every spatial location in a scale-free manner. Using positional encodings, the hypernetwork creates spatially varying hardware patterns that guide the NCA's dynamics. Crucially, this approach maintains spatial invariance—task specifications are normalized relative to grid dimensions, enabling automatic adaptation to different grid sizes and region placements.

This graph-based representation provides an intuitive interface between human intent and NCA computation. Users define tasks through natural graph specifications (inputs, outputs, operations), and the meta-network translates these into appropriate hardware configurations. This effectively creates a true compiler between human intent and the NCA's computational capabilities, allowing for more sophisticated definition of task chaining and temporal hardware evolution.

The graph structure enables rich extensions beyond our initial implementation. Tasks can be ordered through edge attributes for sequential execution planning. Dynamic hardware reconfiguration becomes possible by modifying the task graph over time, creating a secondary dynamics layer complementing the fast neural dynamics of the cellular automaton. This hierarchical temporal structure—fast neural dynamics for computation, slower hardware dynamics for program flow—mirrors traditional computing architectures' separation of clock-cycle operations and program execution, but within a unified differentiable framework.

This approach represents a promising path toward more efficient, adaptable continuous computational paradigms that could fundamentally change how we think about programming NCA-based systems and potentially other emergent computational substrates.

## Conclusion

The exploration of universal computation within cellular automata has historically been confined to discrete systems, where models like Conway's Game of Life and Elementary Cellular Automata have demonstrated the remarkable ability to emulate Universal Turing Machines. However, extending this capability to continuous cellular automata presents significant challenges, primarily due to the absence of discrete states and the inherent instability of smooth, analog dynamics. In this work, we have taken pragmatic first steps toward overcoming these hurdles by leveraging NCA as a substrate for developing universal computation in a continuous domain. By employing gradient descent to train NCA rules, we have demonstrated a pathway to sculpt complex computational behaviors without the need for manual rule design, shifting the burden of discovery from human ingenuity to machine learning.

Our results illustrate that NCA can successfully encode fundamental computational primitives, such as matrix multiplication and inversion, and even emulate a neural network capable of solving the MNIST digit classification task directly within its state. These findings suggest that NCAs can serve as a bridge between traditional computing architectures and self-organizing systems, offering a novel computational paradigm that aligns closely with analog hardware systems. This linkage is particularly promising for designing efficient computational frameworks for AI models operating under constrained resources, where energy efficiency and robustness are paramount. Rather than training entirely new rules for each task, our approach hints at the possibility of discovering optimal hardware configurations that exploit the fixed physical laws governing these substrates, enabling meaningful computations with minimal overhead.

Looking forward, we believe this work lays the groundwork for transformative advancements in computational science. By automating the discovery of general-purpose computers within diverse physical implementations, NCA could revolutionize how we harness novel materials and systems for computation, potentially leading to ultra-efficient analog hardware systems or computational paradigms that scale linearly with resource demands. While challenges remain — such as stabilizing continuous dynamics for reliable symbolic encoding and scaling these systems to more complex tasks—the potential of NCA to unlock universal computation in continuous cellular automata opens new avenues for exploration. Ultimately, this research not only advances our understanding of computation in continuous dynamics but also paves the way for the next generation of adaptive, energy-efficient computing technologies.

<!-- </d-article> -->

<div class="distill-appendix">
  <h3>Citation</h3>
  <p>For attribution in academic contexts, please cite this work as</p>
  <pre><code>Béna, et al., "A Path to Universal Neural Cellular Automata", Genetic and Evolutionary Computation Conference (GECCO '25 Companion), 2025.</code></pre>

  <h3>BibTeX Citation</h3>
  <pre><code>@article{bena2025unca,
    title   = {A Path to Universal Neural Cellular Automata},
    author  = {Béna, Gabriel and Faldor, Maxence and Goodman, Dan and Cully, Antoine},
    journal = {Genetic and Evolutionary Computation Conference (GECCO '25 Companion)},
    year    = {2025},
    doi     = {doi.org/10.1145/3712255.373431}
  }</code></pre>
</div>