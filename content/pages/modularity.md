---
alias:
- modular
title: Modularity
fileName: modularity
tags:
- tag
categories:
- category
date: 2022-11-21
lastMod: 
---


{{renderer :toc_myrfccc}}

  + 

  + # Intro / Definition :

    + The concept of modularity in #neural-networks is not new but has seen a resurgence in recent times.

    + A network is considered modular when it can be broken apart into several #sub-modules,  #sub-modules that should be able to operate independently.
    + With the phenomenal success of #deep-learning in recent years, the gain in performance has often been accompanied with a lack of this modular property in deep #neural-networks.

    + Straying away from overly monolithic architectures towards structurally richer models can however be highly beneficial, as well as being a more accurate representation of biological entities. Modular networks with functionally specialized #sub-networks ( #experts ) are more adaptable, robust and interpretable than entangled monolithic systems.

    + Adaptable because the experts can be composed to face new #tasks created from the combination of old ones, and that new #experts can be added to an existing architecture if needed, to tackle new aspects of a problem.

    + Robust because the disentanglement of function among #experts, and their relative independence from one another, means that the behavior of the whole can sustain malfunction or removal of one of its part.

    + Interpretable because under the right conditions, #experts can naturally lead us to a #decomposition of a global problem into understandable #sub-tasks.

    + I am personally very influenced by Minsky's Society of Mind [[@The society of mind]] , where individually simple and specialized (but mindless) #agents give rise to complex and rich behavior through their interactions.

    + Thus the idea of modularity alone carries with it several key concepts that I find crucial to investigate. These concepts infuse the rest of my research goals as they all naturally combine with this core notion of modularity.

  + # #Ubiquity :

    + Modularity is a broad and complex concept, because it applies to patterns of organization of complex #systems themselves. In that sense, an organized #systems composed of elements and connections can display modularity features when those elements are assembled into relatively independent and autonomous, internally highly connected #clusters.

      + When considering the many forms those elements and connections can take however, we can understand how modularity can become such a vast, yet still uniting, notion. Even with #networks, and more specifically #neural-networks, being the center of focus of my work and this thesis, this notion shouldn't be reduced to purely to a study of #graphs or #networks.

      + More specifically it is to be noted than the study of many systems of interest can be re-framed in that manner. Moreover, the study of any kind of complex systems would seem hopeless if their elements were not assembled into these coherent functional building blocks that we call #modules.

      + It is this fundamental property that allows us to investigate such systems in a higher-level, synthetic point of view [[@Evolution of Complex Modular Biological Networks]]

      + Any organism showing integration and cohesive properties, as well as interactions towards its environment then falls under the banner of modularity studies.

      + This could includes protein-to-protein physical with chemical interactions, cellular assemblies, gene regulation mechanisms through statistical connections and of course #neurons and neuronal connections in animal #brain [[@An introduction to systems biology: design principles of biological circuits]] [[@The road to modularity]].

      + The study of larger scale social network and ecosystems can equally be led under the same point of view. In that regard, the concept unifies a tremendous amount of much older ideas and trends of research. Finding coherent #sub-modules in systems, and understanding their interactions can then be boiled down to a study of networks and communities as was brought to light in the seminal work of [[@Modularity and community structure in networks]]

    + Having now understood how widespread modularity is of a concept, we will focus on its presence and impact in #neural-networks.

      + As in other areas, modularity is ubiquitous in neural architectures, especially in brains of biological organisms. The long sought quest of understanding human cognition and #intelligence has led to the desire of creating and understanding a complete structural-functional mapping of the human brain, with its neuronal elements and connections [[@The Human Connectome: A Structural Description of the Human Brain]]

      + Following this desire, a great amount of work have reported modular architectures in the network of the human brain,  particularly in the #neocortex : [[@Revealing Modular Architecture of Human Brain Structural Networks by Using Cortical Thickness from MRI]] [[@Age-related changes in modular organization of human brain functional networks]] [[@The columnar organization of the neocortex]]

      + Not only is the brain structurally modular, it also exhibits modular properties in a functional point of view. Function has been shown to be highly distributed and disentangled in the brain, with certain areas being specialized into certain cognitive functions. This was confirmed using fMRI and other techniques, as well as by studying brain injuries and lesions.

      + Graph theory was later introduced to study these properties and the brain was shown to exhibit a small world nature both in #structure and #function [[@Age-related changes in modular organization of human brain functional networks]]

      + The average shortest-path between nodes grows slowly (logarithmically) as a function of the number of nodes on the network, but the network is still highly clustered.

      + That implies that even if most nodes are not neighbors of one another, you can reach most nodes from most others in a small number of steps [[@Encyclopedia of Complexity and Systems Science]] .

    + This property is very beneficial in terms of information transfer and processing. But why is it that modularity appears everywhere ?

  + # Origins :

    + If modularity is that widespread in biological systems, it should show evolutionary advantages that led to this ubiquity.

    + One hypothesis is that modularity naturally emerges when goal-oriented systems are placed in an environment with varying objectives. More importantly this is the case when overall goals change over-time while sharing some of their sub-problems with prior objectives.

      + Such environments are called modularly varying goals (MVG) and have been shown to speed up evolution of networks (the speed at which they are able to achieve a given goal), as well as promote evolvable and modular solutions [[@Varying environments can speed up evolution]]

      + This assumption seems reasonable as natural environments indeed would fit under this kind of goal-changing assumption. We will go back to that idea a bit later, when talking about notions of #continual-learning and see how modularity certainly can be very advantageous in tackling these type of problems. However, if this is certainly part of the global answer, it had also been shown not to explain all of it [[@Specialization Can Drive the Evolution of Modularity]]

    + Another idea postulates that modularity could evolve under the pressure of minimizing connections costs between the units of a systems :

      + Most biological systems come with some kind of cost for creating and maintaining connections between its different parts, as well as sending messages through said connections. These costs all depend on the overall connection number and lengths in a network.

      + It has been shown that the summed length of the wiring in the brain have been minimized, which supports this idea of evolutionary pressure [[@The Wiring Economy Principle: Connectivity Determines Anatomy in the Human Brain]] .

      + Researchers have then speculated than modularity could evolve as a byproduct of this cost minimization. This idea was reported in theoretical neuroscience [[@Principles of brain evolution]] and then confirmed in evolutionary biology experiments [[@The evolutionary origins of modularity]] [[@Neural Modularity Helps Organisms Evolve to Learn New Skills without Forgetting Old Skills]] .

        + In this particular work it was shown that evolving neural networks to solve a task under a multi-objective (performance + connection-cost) fitness function not only leads to more modular solutions, but also to better performances overall.

        + We can then imagine than what would have begun as a cost optimization led to several other evolutionary advantages, resulting in modularity being maintained and thus ending ubiquitous.

    + Let us examine those advantages in more detail :

  + # Advantages :
    + 

    + ## #Robustness :

      + How brain regions, or other complex networks for that mater, can display such robustness against perturbation and corruption is still an open question.

      + Studies have shown that two properties are however essential to achieve it : modularity and redundancy.

      + Modularity can help robustness by design : Because sub-modules are loosely coupled and fairly independents, the malfunction or error of one of the parts of a modular neural network does not imply a general dysfunctionment.

      + Even better, this makes it easier to locate and solve any arising errors, by locating the module the error took place in, or the dysfunctional objective carried by said module.

      + Redundancy on the other hand, prevents sub-modules carrying crucial computation to be irreversibly destroyed or corrupt, and allow information to be recovered when needed.

      + As such, a network learning redundant modular representation becomes extremely robust [[@Modularity and robustness of frontal cortical networks]]

    + ## #Adaptability :

      + This idea is related to the concepts of #continual-learning and #catastrophic-forgetting (or interference).

      + In many cases training is solely focused on achieving performance on the latest objective and functionality is entangled in the network. Then when learning a new task, most of the knowledge acquired in prior tasks (usually stored in the weights) of a network will be overwritten.

      + In that regard, new information interferes with previously learned knowledge [[@Continual Lifelong Learning with Neural Networks: A Review]]

      + Modularity is able to mitigate that effect by distributing functionality among different independent #modules, for which learning can be activated or not.

      + From an engineering point a view, when a new task requires to be learned, a different #module can either be used, or a new one added, without touching on previously trained modules.

      + Modularity have been successfully applied to #continual-learning scenarios in several lines of work :

        + [[@Neural Modularity Helps Organisms Evolve to Learn New Skills without Forgetting Old Skills]]

        + [[@Supermasks in Superposition]] [[@PathNet: Evolution Channels Gradient Descent in Super Neural Networks]]

        + [[@ROUTING NETWORKS AND THE CHALLENGES OF MODULAR AND COMPOSITIONAL COMPUTATION]]

        + [[@ROUTING NETWORKS: ADAPTIVE SELECTION OF NON-LINEAR FUNCTIONS FOR MULTI-TASK LEARN-]]

    + ## #Compositionality :

      + Building on this idea of continual learning, comes the concept of #compositionality. The knowledge we constantly acquire isn't composed of separate and isolated objectives and ideas.

      + On the contrary, every new task fit in a large graph of relations between objectives. These goals sometimes share sub-problems, and sometimes can be recovered by composing other goals.

      + This idea contrast with the previous one -the old knowledge shouldn't interfere with newly acquired one- by stating that it should actually guide it.

      + We can see this as a positive versus negative transfer problem, or the transfer-interference trade-off [[@ROUTING NETWORKS AND THE CHALLENGES OF MODULAR AND COMPOSITIONAL COMPUTATION]] [[@Recursive Routing Networks: Learning to Compose Modules for Language Understanding]]

      + Solving this problem is particularly tricky, and engages ideas from #continual-learning, #meta-learning and #transfer-learning.

      + It is however crucial to achieve compositional (or combinatorial) generalization, one of the most amazing and powerful feature of human's cognition and reasoning, allowing us to make "infinite use of finite means" [[@On Language: On the Diversity of Human Language Construction and its Influence on the Mental Development of the Human Species]] .

      + This could be the key to achieve machine intelligence that understands concepts and models rather that only recognizes patterns [[@Building Machines That Learn and Think Like People]]

    + ## #Efficiency :

      + Finally, modular networks gain efficiency by allowing only some of their parts to be active when need it be. This also means that resource consumption can scale up dynamically with demand, with more complex tasks involving more modules being attributed more resources.

  + # #Modular #neural-networks

    + Having examined the concept of modularity, what it implies and from where it comes from, we can now take a look at some practical experiments and implementations of modular neural networks (MNNs) that have been done and are relevant to my work. In their review, \cite{Amer_2019} describes how modularity can be achieved in 4 specific areas or techniques :

    + When discussing existing MNNs I will try and refer to those 4 points to better understand and classify the different approaches. One point in particular seemed to neatly separate two possible approaches to modular networks : topology and how modularity is structurally achieved. Modularity can be either directly hand-crafted into the network by design, or achieved through learning and evolution.

    + ## Implicit Modularity

      + ### Evolutionnary Approach :

        + When discussing the origin of modularity, I referred to two important works :

        + [[@The evolutionary origins of modularity]] [[@Neural Modularity Helps Organisms Evolve to Learn New Skills without Forgetting Old Skills]]

        + These contributed to understand how modularity can emerge as an evolutionary process. In these works the authors evolve neural networks, meaning weights are determined via an evolutionary algorithm at the beginning of an network's 'lifetime'.

        + In the first work, weights are then fixed during the whole lifetime whereas in the second one a learning phase follows the evolution phase. The main result of [[@The evolutionary origins of modularity]] is that evolving neural networks under a connection-cost pressure leads to more modular architecture. The task they examine consists of two separate images and is thus modular in itself (Domain). The architecture isn't hand crafted to be modular but rather evolve to be during the formation phase.

        + The work in [[@Neural Modularity Helps Organisms Evolve to Learn New Skills without Forgetting Old Skills]] then builds on that idea showing that using this connection-cost pressure to achieve modular architectures, also proves beneficial in varying goal environment. In fact that modular solutions retain prior knowledge better and suffer less from catastrophic interference when confronted with varying objectives. This approach to modularity is very interesting since it is achieved as an emergent property, rather than a handcrafted feature like in many other work.

      + ### #Weight-Masks :

        + Another interesting concept resides in  the idea of weight-masking. Having an existing network with its weights already trained (or not) and fixed, it is possible to train and apply binary masks on those weights to discover which part of the network accounts for which computation of the final objective.

        + This weight level analysis of a network can reveal modularity after a network has been trained [[@ARE NEURAL NETS MODULAR? INSPECTING FUNC- TIONAL MODULARITY THROUGH DIFFERENTIABLE WEIGHT MASKS]] or find high performing sub-networks in randomly initialized networks [[@THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS]] / [[@What’s Hidden in a Randomly Weighted Neural Network?]]

        + Then [[@Supermasks in Superposition]] / applied that idea to successfully train masks accounting for thousands of different objectives on the same random and over-parametrized network. These masks constitutes functionally separate modules, but it remains to be seen if they could be structurally separate as well.

        + I am currently trying to reproduce the results concerning connection-cost minimization on a larger scale. I am also investigating this idea of masking on an larger level, on a network containing multiple explicitly designed sub-modules, and masks being applied on a module level.

      + In the following works, modularity will largely be handcrafted and weights are usually learned using backpropagation.

    + ## Explicit Structural Modularity :

      + ### #Routing :

        + Routing is a powerful idea that allows the composition of different modules into a coherent ensemble. A routing network is composed of a set of modules, from which a router can assemble a network using a composition strategy.

        + [[@ROUTING NETWORKS AND THE CHALLENGES OF MODULAR AND COMPOSITIONAL COMPUTATION]] [[@ROUTING NETWORKS AND THE CHALLENGES OF MODULAR AND COMPOSITIONAL COMPUTATION]]

        + [:span]
ls-type:: annotation
hl-page:: 2


        + Training a routing network involves jointly training their modules and the module-composition policy. In that sense it tackles modularity in 3 of the phases described earlier : architecture, formation and integration. That joint training however creates a lot of challenges as well :

        + Something that can happen is 'module collapse', where the same modules are selected over and over during composition thus destroying the potential flexibility and dimensionality of the network.

        + On the other hand, this added flexibility can result in severely overfitting networks as well. [[@ROUTING NETWORKS AND THE CHALLENGES OF MODULAR AND COMPOSITIONAL COMPUTATION]] address this issue as the 'flexibility dilemma' showing that both issues are related to one spectrum of effects. A router with low flexibility using only handful of modules leads to underfitting, whereas a too flexible router will overfit badly, both resulting in bad generalization.

      + ### Diversity/Heterogeneity :


        + The question of module diversity is also of crucial importance in routing networks, providing the possibility for even more flexibility and power through this diversity.

        + Heterogeneity is crucial in many aspects of complex systems, from neurons themselves [[@Neural heterogeneity promotes robust learning]] / to modules in a modular architecture.

        + In [[@WILDA: Wide Learning of Diverse Architectures for Classification of Large Datasets]] the authors achieve good performance by initializing a set of very architecturally different modules using the genetic algorithm #MAP-Elites [[@Illuminating search spaces by mapping elites]] /, and combining them into an wide ensemble model. In this case the topology and formation are determined by this evolutionary process. Integration then relies on a single shallow network that gets inputs from this diverse set of prior modules.

        + Taking almost the opposite approach, [[@Encoders and Ensembles for Task-Free Continual Learning]] use a single pre-trained classifier to encode inputs, and then relies on an ensemble of simple classifiers for integration. Each classifier produces a key in the same dimension space that the encoded input. Classifiers are then picked using a k-nearest-neighbors algorithm, ensuring that no task specification is needed to select the appropriate output layer :

        + [:span]
ls-type:: annotation
hl-page:: 3


        + 

      + ### Decomposition :


        + Sub-network in a modular architecture would ideally decompose #functionality, by having different modules attend to different aspects of the global task. This decomposition however can be a bit more subtle and can carry some crucial biases, as in [[@Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer]]  : [:span]
ls-type:: annotation
hl-page:: 2


        + They prove that this modular decomposition allow for sharing task information across robots, and reciprocally robot information across tasks. They are then able to recompose these modules in a 'mix and match' fashion to perform on a task-robot ('world') combination never seen before during training.

        + A similar approach is tried out in the framework of supervised learning in [[@Towards Reusable Network Components by Learning Compatible Representations]] . In that case decomposition is achieved by having separate modules for feature extraction and task-specific classification. They manage to ensure that these modules are trained in a compatible manner, resulting in components re-usable and composable.

      + ### #Attention :


        + Finally, attention mechanisms provide a way to successfully train and integrate different modules into a coherent ensemble.

        + In [[@Coordination Among Neural Modules Through a Shared Global Workspace]] [[@RECURRENT INDEPENDENT MECHANISMS]] , authors design a modular architecture composed of recurrent sub-networks competing via attention mechanisms to communicate : [:span]
ls-type:: annotation
hl-page:: 2


        + This effectively create a communication bottleneck, which might be required for #[functional-specialization]({{< ref "/pages/functional-specialization" >}}), resulting in a linear scaling in terms of number of agents (compared to a quadratic scaling when using pairwise interaction).

        + Authors also argues that this competition endow the model with a coherent 'common language' between specialists, idea first brought up by [[@What is consciousness, and could machines have it?]] [[@Towards a cognitive neuroscience of consciousness: basic evidence and a workspace framework]] The cited advantages of modularity # Advantages :
 are confirmed with authors showing great generalization and robustness of their architecture, as well as efficiency with the linear communication scaling.
