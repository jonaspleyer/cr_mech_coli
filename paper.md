---
title: 'cr_mech_coli: A mechanical Agent-Based Model of Elongated Bacteria for Parameter Estimation and Data Generation'
tags:
  - python
  - biology
  - agent-based
  - cellular
authors:
  - name: Jonas Pleyer
    orcid: 0009-0001-0613-7978
    affiliation: 1
  - name: Moritz Steinmaier
    orcid:
    affiliation: 2
  - name: Jelena Bratulić
    orcid: 0009-0007-2835-4168
    affiliation: 2
  - name: Thomas Brox
    orcid: 0000-0002-6282-8861
    affiliation: 2
  - name: Christian Fleck
    affiliation: 1
    orcid: 0000-0002-6371-4495
affiliations:
  - index: 1
    name: Freiburg Center for Data Analysis and Modelling and AI (FDMAI), University of Freiburg, Freiburg, Germany
    ror: 0245cg223
  - index: 2
    name: Computer Vision Group, Faculty of Engineering, University of Freiburg, Freiburg, Germany
    ror: 0245cg223
date: 13 March 2025
bibliography: paper.bib
header-includes:
  - '`\usepackage{tabularx}`{=latex}'
  - '`\usepackage{booktabs}`{=latex}'
  - '`\usepackage{xcolor}`{=latex}'

---

# Summary
Constructing numerical simulations of biological systems is a challenging problem commonly addressed
using Agent-Based Models (ABMs) [@Pleyer2023;@Ghaffarizadeh2018].
We present `cr_mech_coli`, a Python package providing a mechanistic Agent-Based Model for elongated,
flexible rod-shaped bacteria, such as _E.Coli_ or _B.subtilis_.
It bridges the gap between numerical Agent-based simulations and microscopic image data by providing
parameter estimation functionalities and data generation capabilities.
Our software builds upon `cellular_raza` [@Pleyer2025] and comprises a computational model, data
extraction techniques, predefined cost functions, and visualization via realistic synthetic
microscopic rendering, as well as other modular components for the parameter estimation workflow.
In particular, `cr_mech_coli` is able to generate synthetic labeled microscopic images from any
simulation output, which can be used to train machine learning models targeting cell segmentation
and tracking.
This package closes the gap between agent-based modeling, model validation & calibration and data
synthesis.

# Statement of Need
Agent-Based Models (ABMs) provide a natural lens to map biological systems to numerical simulations
[@Pleyer2023;@Ghaffarizadeh2018;@Cooper2020;@Karolak2021;@DeRybel2014].
They express cellular behaviour in functions of individual agents.
Although existing tools study various cell types, including spherical, hexagonal, and cylindrical
[@Young2006;@Young2007], they mostly lack the ability to model flexible, elongated bacteria, such
as _B.subtilis_.
Moreover, to validate the computational model against experimental data, parameter estimation
techniques to calibrate the model [@Kreutz2013;@Raue2014] are required, which have not yet been
applied in these scenarios.
This significantly limits the confidence in such Agent-Based Models to accurately describe the
underlying biological reality.

There is currently no systematic approach to estimating agent-level parameters at the single-cell
level within an agent-based modeling framework.
Existing calibration attempts [@An2016;@Lima2021;@Dancik2010;@Thiele2014] often rely on population
data by aligning the distributions of readouts obtained from the chosen target system instead of
estimating parameters on the cellular level.
They also fail to properly capture the individual nature of these systems, including effects such 
as cellular heterogeneity.
This leaves gaps regarding the interpretability of which parameters drive the observed phenomena.

Another challenge lies in generating synthetic, realistic-looking data that extends beyond visual
appearance to include accurate cellular dynamics.
Recently, diffusion-based models have demonstrated strong performance on static configurations
[@Han2025;@eschweiler2024celldiffusion;@Sturm2024 ].
However, they still struggle with the temporal domain and with ensuring consistency in modeling
mechanistic properties alongside the cells' visual appearance.
Moreover, recent studies on video diffusion models report systematic failures in physical reasoning
and in modeling consistent physical dynamics, even when trained at scale
[@kang2024how;@motamed2025generative].

Our `cr_mech_coli` addresses this methodological gap by providing a complete framework as a
user-friendly Python package.
It provides a predefined mechanical model for such elongated bacteria, often referred to as rods,
which can be used in 2D and 3D scenarios.
To initialize such a model from microscopic images, we also provide methods to extract positional
information for agents from masks generated from these images.
Furthermore, to compare numerical simulation results with microscopic images, we provide a variety
of methods that either compare the underlying cellular representation as a set of vertices or
generate new cell masks which can be used in traditional image-comparison schemes.
Our software operates in a modular and generalizable fashion, allowing researchers to use it across
a variety of models and estimation techniques.
While this publication focuses on the software, we are concurrently preparing another submission
that provides a more detailed description of the mathematical model and applies these methods to a
collection of case studies.

# Software design
Our framework comprises 4 interleaved components:

1. A computational model that simulates bacterial behaviour
2. Methods to generate instance-level cell masks
3. Parameter estimation methods for fitting a computational model to data
4. Visualization methods for rendering realistic synthetic microscopic images

## Computational Model
\autoref{table:simulation-aspects} contains a list of the simulated aspects of bacterial behaviour.
We represent the bacteria as a collection of vertices $\{\textbf{x}_i\}$ which can be viewed as a
discretization of the bacterium (\autoref{fig:model-vertices-interaction} (A)).
The vertices are connected via springs with fixed lengths.
Growth is simulated by gradually increasing the size of these segments.
To describe the flexibility of the rods, we use a bending force describing a bent beam, which is
applied at each vertex individually.
The interaction between bacteria can be described by a Morse or a Mie interaction potential, which
generalizes the popular Lennard-Jones potential.
Interactions are calculated from each vertex to the nearest point on the polygon of the interaction
partner.
This is displayed in more detail in \autoref{fig:model-vertices-interaction} (B).
Our model takes a generalist stance, allowing researchers to build on top of it and making it
applicable to a wide range of bacterial systems.

\begin{table}[!h]
    \centering
    \def\arraystretch{1.3}
    \begin{tabularx}{\textwidth}{c l X}
        &\textbf{Aspect} & \textbf{Description}\\
        \toprule
        &\multicolumn{2}{l}{\textbf{(C) Cellular}}\\
        \midrule
        (1) & Rod-Shaped Mechanics &
            Rod-shaped bacteria are flexible rods which are able to freely move around (off-lattice
            approach)
            (\protect\hyperlink{ref-Amir2014_2}{Amir et al., 2014};
            \protect\hyperlink{ref-Takeuchi2005}{Takeuchi et al., 2005};
            \protect\hyperlink{ref-Ursell2014}{Ursell et al., 2014})\\
        (2) & Growth &
            Cells grow exponentially by inserting new material either along the circular part of the
            rod or at the tip
            (\protect\hyperlink{ref-Robert2014}{Robert et al., 2014};
            \protect\hyperlink{ref-Takeuchi2005}{Takeuchi et al., 2005})\\
        (3) & Division &
            Rod-Shaped bacteria divide in the middle of the rod into two new agents.\\
        &\multicolumn{2}{l}{\textbf{(CC) Cell-Cell Interactions}}\\
        \midrule
        (4) & Adhesion &
            Bacteria attract at longer distances and adhere to each other at shorter
            distances
            (\protect\hyperlink{ref-Trejo2013}{Trejo et al., 2013};
            \protect\hyperlink{ref-Verwey1947}{Verwey, 1947})\\
        \bottomrule
    \end{tabularx}
    \caption{
        This table lists aspects of cellular behavior which we consider in our computational model.
        It is split into cellular aspects (C) which concern only an individual bacterium and
        interactions between cells (CC).
        These aspects are generalized in order to fit to as many types of rod-shaped bacteria as
        possible.
        These are the assumptions for the basic model that we are presenting here.
        In order for our methods to work, we are really only required to satisfy the assumption of
        our spatial representation and thus, these assumptions can also be loosened if required.
    }
    \label{table:simulation-aspects}
\end{table}

\begin{figure}[!h]
    \centering
    \begin{minipage}{0.5\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/mechanics.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{A}
        \vspace*{\textwidth}
    \end{minipage}%
    \begin{minipage}{0.5\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/interaction.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{B}
        \vspace*{\textwidth}
    \end{minipage}
    \caption{
        (A) Schematic representation of a bacterial agent consisting of vertices which are connected by
        springs.
        (B) Agents interact by calculating forces between each vertex and the nearest point on the
        polygon of the interaction partner.
        These connections are depicted in orange lines.
    }
    \label{fig:model-vertices-interaction}
\end{figure}

## Parameter Estimation
To estimate the parameters of the computational model, we require methods that enable comparison of
experimental data with numerical outputs.
Furthermore, we need to be able to initialize our model with experimental data by specifying values
of the discretization vertices $\{\textbf{x}_i\}$.
And finally, numerically obtained results need to be compared with experimental data to facilitate
parameter estimation techniques.

### Data Extraction & Direct Comparison
We utilize the discretization in vertices $\{\textbf{x}_i\}$.
We assume that the given microscopic image (such as in \autoref{fig:parameter-estimation} (A))
has already been segmented by a fitting segmentation tool [@Cutler2022;@Stringer2020;@Hardo2022].
Using the `scikit-image` package [@vanderWalt2014], we perform a skeletonization for each individual
cell-submask @Lee1994 (see \autoref{fig:parameter-estimation} (B)).
Afterwards, we extract the individual vertices from the skeleton (\autoref{fig:parameter-estimation}
(C)).
These positions can now be used to initialize the agents correctly in space as well as for comparing
numerical outputs to data.

### Using Instance-Level Masks for Imaging-Based Comparison
The extraction algorithm enables comparison of cellular states across intervals without division
events.
However, this technique can not be applied when division events change the number of present
vertices, since this would require comparing arrays of unequal dimensions.
We can, therefore, realize another approach in which we render cell masks of the numerically
obtained results and compare the resulting image to the experimental cell masks.
This technique does not require any assumptions about the agents' underlying spatial representation.
To properly compare cellular states, the cell lineage of the data needs to be mapped to the
numerical results.
We use an in-image encoding scheme that assigns instance-level segmentation masks, where each
instance (specific cell) is paired with a unique persistent color throughout the entire simulation.
\autoref{fig:parameter-estimation} (D,E) shows two images of differing configurations that are being
compared to each other.
When only comparing instance masks directly, we arrive at subfigure (F), where differences are
highlighted in white and matches are black.
In practice, the time of the division event is often not precisely determined, and segmentation
tools are prone to mislabeling already divided bacteria. Therefore, we account for the parental
relationship between the bacteria when comparing different cellular configurations.
We adjust the assigned cost value when the two instances are in a mother-daughter relationship, as
shown in \autoref{fig:parameter-estimation} (G). The differences in parental relationships between
the cells are indicated in gray.

\begin{figure}[!h]
    \centering
    \begin{minipage}{0.36\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/fitting-methods/algorithm/image001042.png}%
        \vspace*{-0.75\textwidth}
        \hspace*{0.5em}\textbf{\color{white}A}
        \vspace*{0.75\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.36\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/fitting-methods/algorithm/mask-zoom.pdf}%
        \vspace*{-0.75\textwidth}
        \hspace*{0.5em}\textbf{\color{white}B}
        \vspace*{0.75\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.255\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/fitting-methods/algorithm/interpolate-positions.pdf}%
        \vspace*{-1.059\textwidth}
        \hspace*{0.5em}\textbf{\color{white}C}
        \vspace*{1.059\textwidth}
    \end{minipage}
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/fitting-methods/progressions-1.pdf}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}D}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/fitting-methods/progressions-2.pdf}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}E}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/fitting-methods/progressions-3.pdf}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}F}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/fitting-methods/progressions-4.pdf}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}G}
        \vspace*{\textwidth}
    \end{minipage}
    \caption{
        (A) Microscopic image of 6 \textit{E.Coli} used as an initial datapoint.
        (B) Segmentation mask of (A) with the extracted skeleton overlaid.
        (C) Extrapolated vertices from the skeleton of the bacterium from (B).
        (D-E) Snapshots of two synthetic masks that are being compared.
        (F-G) Difference of (D) to (E). Matching colors are black, differences are white.
        (G) additionally includes parental relationships, indicating that one cell is the daughter of the other, shown in gray.
    }
    \label{fig:parameter-estimation}
\end{figure}

### Scripts
We combine the computational model with the data extraction techniques to estimate the parameters of
our model.
To this end, we provide a predefined script `crm_fit` which is able to automatically perform this
task.
It can automatically generate likelihood profiles of the estimated parameters, as shown in
\autoref{fig:likelihood-profiles} (A-C).
Furthermore, we provide exemplary scripts `crm_amir` and `crm_divide` to show how `cr_mech_coli`
can be used in particular settings, the former containing a model of a flexible elongated rod and
the latter including division events.
Our methods are designed in such a way that standardized tools can be used in tandem with
`cr_mech_coli`.
We encourage readers to explore the code of the `crm_amir` script as an initial starting point for
parameter estimation.

\begin{figure}[!h]
    \centering
    \begin{minipage}{0.3\textwidth}
        \includegraphics[width=\textwidth]{figures/damping.pdf}%
        \vspace*{-\textwidth}
        \textbf{A}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.3\textwidth}
        \includegraphics[width=\textwidth]{figures/strength.pdf}%
        \vspace*{-\textwidth}
        \textbf{B}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.3\textwidth}
        \includegraphics[width=\textwidth]{figures/radius.pdf}%
        \vspace*{-\textwidth}
        \textbf{C}
        \vspace*{\textwidth}
    \end{minipage}%
    \caption{
        Exemplary likelihood profiles for (A) stokesian damping constant, (B) strength of the
        interaction potential and (C) radius (thickness) of the bacterial rod.
        These profiles can be generated by estimating the parameters of the provided model with
        experimental microscopic images.
        Afterwards, each individual parameter is scanned within its optimization interval while the
        remaining parameters are reoptimized.
    }
    \label{fig:likelihood-profiles}
\end{figure}

## Visualization & Data Generation
Our framework further supports visualization of the computational model's output, providing a
resource-efficient and scalable method for generating high-quality segmentation and tracking data
for training deep learning models.
Visualization builds on representing cellular structures as meshes in PyVista
[@sullivan2019pyvista], as shown in \autoref{fig:pipeline} (A), where each rod-shaped bacterium is
constructed by merging a series of spheres and cylinders into a continuous, smooth mesh.
Afterwards, these 3D models are projected along the z-axis, producing raw intensity images
(\autoref{fig:pipeline} (B)) and corresponding ground-truth segmentation masks
(\autoref{fig:pipeline} (D)).
However, the resulting raw-intensity images do not resemble realistic microscopic data, missing
optical effects and distortions such as sensor noise, halo effects, and cellular texture.
Thus, to minimize the domain gap between synthetic and empirical data, the raw projection is
processed through a physics-inspired imaging pipeline, simulating the specific distortions and
artifacts inherent to real-world microscopy, as shown in (\autoref{fig:pipeline} (C)).
We apply a multi-layered transformation sequence:

- Environmental Modeling: We construct a spatially varying background by integrating multi-scale
  noise, illumination gradients, and simulated debris.
- Optical Artifacts: Characteristic phase-contrast halos are generated using distance-transform
  masks with configurable attenuation, whereas optical diffraction is modeled via point spread
  function (PSF) convolution.
- Sensor Noise: To replicate digital acquisition artifacts, we inject signal-dependent Poisson
  (shot) noise and Gaussian (readout) noise.

To represent different configurations of various microscopic apparatuses, we adopt an automated
parameter-fitting routine based on differential evolution [@storn1997differential].
The optimization procedure tunes the simulation's core imaging parameters against real reference
images (\autoref{fig:pipeline} (E)) by minimizing a weighted objective function of Structural
Similarity (SSIM), Peak Signal-to-Noise Ratio (PSNR), and histogram distance.
To this end, we extract the positional vertices $\{\textbf{x}_i\}$ from the provided cell masks
(\autoref{fig:pipeline} (G)) and estimate the thicknesses of the bacteria in order to initialize the
rendering engine.
The resulting image (\autoref{fig:pipeline} (F)) then closely resembles the provided input image.
Using the optimized visualization parameters, we are now able to generate arbitrary amounts of data
by using our mechanistic model.

\begin{figure}[!h]
    \centering
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/imaging-mesh.pdf}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{A}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{figures/raw_rendering.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}B}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{figures/final_image.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}C}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.24\textwidth}
        \includegraphics[width=\textwidth]{figures/segmentation_mask.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}D}
        \vspace*{\textwidth}
    \end{minipage}
    \begin{minipage}{0.32\textwidth}
        \includegraphics[width=\textwidth]{figures/original_CEX.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}E}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.32\textwidth}
        \includegraphics[width=\textwidth]{figures/synthetic_CEX.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}F}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.32\textwidth}
        \includegraphics[width=\textwidth]{figures/original_CEX_masks.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}G}
        \vspace*{\textwidth}
    \end{minipage}%
    \caption{
        (A) Cells are rendered by combining spheres and cylinders into a single mesh.
        Spheres are highlighted in blue and cylinders in gray.
        (B) A result from combining sphere and cylinder meshes to obtain the shape of a bacterium.
        (C) Rendered image after applying photo-realistic filters.
        (D) Instance-level cell masks with unique color assignment to each agent.
        (E) Real microscopic image of the M90T strain of \textit{Shigella flexneri} taken from the
        omnipose data set (\protect\hyperlink{ref-Cutler2022}{Cutler et al., 2022}).
        (F) Synthetically generated microscopic image with optimized parameters.
        (G) Instance-level cell masks corresponding to (E).
    }
    \label{fig:pipeline}
\end{figure}

# Research impact statement
`cr_mech_coli` provides a user-friendly pipeline for simulating bacterial populations and offers a
reliable, easily modifiable framework for researchers studying bacterial growth and its properties.
A realistic synthetic microscopic image generation pipeline can be leveraged to produce high-quality
labeled segmentation masks and tracking ground-truth data for training a deep learning model,
thereby addressing a key bottleneck in quantitative microbiology and AI-assisted biochemistry
applications, where manually annotated ground-truth data are scarce and expensive to obtain.

# AI usage disclosure
We used Claude Opus [@claude_opus2025] as a coding assistant during the implementation of the data
generation functionality encapsulated within the `crm_imaging` module and `crm_gen` scripts.
All other code-related implementations were done without the assistance of AI systems.
Further, we used Grammarly [@grammarly] to polish the writing, while all core contributions and the
initial draft were done by the authors.

# Acknowledgements
JB acknowledges support from the German Research Foundation (DFG) under grant 499552394
(SFB 1597 - Small Data).

## Author Contributions
**Conceptualization:** Jonas Pleyer, Jelena Bratulić, Moritz Steinmaier\newline
**Software:** Jonas Pleyer, Moritz Steinmaier\newline
**Writing:** Jonas Pleyer, Jelena Bratulić, Moritz Steinmaier\newline
**Supervision:** Christian Fleck, Thomas Brox

# References

