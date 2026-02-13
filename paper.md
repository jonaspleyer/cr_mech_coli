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
`cr_mech_coli` is a Python package with an Agent-Based Model of flexible, elongated bacteria which
provides extensive methods to fit this model to microscopic data.
The package consists of the computational model, visualization methods, data extraction techniques
and predefined cost functions as well as other components which can be combined modularly to enable
parameter estimation workflows.
It is backed by a Rust codebase which builds on `cellular_raza` [@Pleyer2025].
This package aims to tighten the gap between agent-based modeling and classical model validation
and calibration techniques.

# Statement of Need
Agent-Based Models (ABMs) provide a natural lens to map biological systems to numerical simulations
[@Pleyer2023;@Ghaffarizadeh2018;@Cooper2020;@Karolak2021;@DeRybel2014].
They express cellular behaviour in functions of individual agents.
Although a plethora of tools exist to study various types of cells [@Young2006;@Young2007] such as
spherical, hexagonal and cylindrical they mostly lack in their ability to describe flexible
elongated bacteria.
Moreover in order to validate the computational model against experimental data, parameter
estimation techniques [@Kreutz2013;@Raue2014] to calibrate the model at hand are required which have
so far not been applied in this scenario.
This significantly limits the ability of such agent-based models to accurately describe the
underlying biological reality.

There is currently no systematic effort to properly estimate parameters of agents on a single-cell
level within an agent-based modeling framework.
Existing calibration attempts [@An2016;@Lima2021;@Dancik2010;@Thiele2014] often rely on population data
by aligning distributions of obtained readouts of the chosen target system.
They therefore fail to properly capture the intrinsic cellular heterogeneity of these systems.
This leaves gaps with respect to the interpretability of which parameters are the driving factors
for the observed phenomena.

To address this methodological gap, our crate `cr_mech_coli` provides a complete framework.
It provides a pre-defined mechanical model for such elongated bacteria, often referred to as rods
which can be used in 2D and 3D scenarios.
In order to initialize such a model from microscopic images, we also provide methods to extract
positional information of such agents from masks generated from microscopic images.
Furthermore, to compare numerical simulation results with microscopic images, we provide a variety
of methods that can either compare the underlying cellular representation as a collection of
vertices or generate new cell masks which can then be used for comparison.
The software is built in a modular, generalized fashion, thus allowing researchers to be reused
across a variety of models and with different estimation techniques.
It is accompanied with an external publication which gives a more detailed description
of the mathematical model and applies these methods to a collection of case-studies.

# Internals
## Computational Model
\autoref{table:simulation-aspects} contains a list of the simulated aspect of bacterial behaviour.
We represent the bacteria as a collection of vertices $\{\textbf{x}_i\}$ which can be viewed as a
discretization of the bacterium (\autoref{fig:model-vertices-interaction} (A)).
The vertices are connected via springs with fixed lengths.
Growth is simulated by gradually increasing the size of these segments.
To describe the flexibility of the rods, we use a bending force describing a bent beam which is
applied at each vertex individually.
The interaction between bacteria can be described by a Morse or a Mie interaction potential which
generalizes the popular Lennard-Jones potential.
Interactions are calculated from each vertex to the nearest point on the polygon of the interaction
partner.
This is displayed in more detail in \autoref{fig:model-vertices-interaction} (B).

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
        our spatial representation (see \nameref{subsec:spatial-representation}) and thus, these
        assumptions can also be loosened if required.
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

## Visualization & Data Generation
We use `pyvista` [@sullivan2019pyvista] to render our results as 3D objects.
As \autoref{fig:image-generation} (A) shows, the rod is visualized by combining a series of spheres
and cylinders into a smooth mesh.
This resulting intermediate representation can then either be rendered as an artistic visualization
of the system (\autoref{fig:image-generation} (B)) or we can project the image along the z-axis,
which enables us to generate cell masks (\autoref{fig:image-generation} (C)).
There is ongoing work on how to render realistic microscopic images such that deep learning
frameworks for cell segmentation and cell tracking can be fed with the combination of realistic
images and exact cell masks.
We also utilize the `plotters` library [@Erhardt2024] for the strictly 2D case in which we can
simplify the visualization process and achieve significant speedups.

\begin{figure}[!h]
    \centering
    \begin{minipage}{0.32\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/imaging-mesh.pdf}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{A}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.32\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/11571737453049821261/raw_pv/000000400.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}B}
        \vspace*{\textwidth}
    \end{minipage}%
    \hspace{0.01\textwidth}%
    \begin{minipage}{0.32\textwidth}
        \includegraphics[width=\textwidth]{docs/source/_static/11571737453049821261/masks/000000400.png}%
        \vspace*{-\textwidth}
        \hspace*{0.5em}\textbf{\color{white}C}
        \vspace*{\textwidth}
    \end{minipage}%
    \caption{
        (A) Cells are rendered by combining spheres and cylinders into a single mesh.
        Spheres are highlighted in blue and cylinders in gray.
        (B) A result from combining sphere and cylinder meshes to obtain the shape of a bacterium.
        (C) Unique color values are assigned to each agent and additional lighting effects are
        removed which ensures that the final image only contains one singular color per cell and
        black background.
        Finally, the image is rendered by projecting along the z-axis.
    }
    \label{fig:image-generation}
\end{figure}

## Parameter Estimation
In order to be able to estimate parameters of the computational model, we require methods that allow
us to compare experimental data to numerical outputs.
To achieve this we utilize the discretization in vertices $\{\textbf{x}_i\}$.
We assume that the given microscopic image (such as in \autoref{fig:parameter-estimation} (A))
has already been segmented by a fitting segmentation tool [@Cutler2022;@Stringer2020;@Hardo2022].
Using the `scikit-image` package [@vanderWalt2014], we perform a skeletonization for each individual
cell-submask with the method developed by @Lee1994 (see \autoref{fig:parameter-estimation} (B)).
Afterwards, we extract the individual vertices from the skeleton (\autoref{fig:parameter-estimation} (C)).
These positions can now be used to initialize the agents correctly in space as well as for comparing
numerical outputs to data.

The aforementioned technique can not be applied when division events change the number of present
vertices since this would require comparing the arrays of uneven dimension.
We can therefore realize another approach which is also void of the assumption about the underlying
discretization of the agents.
That is to render cell masks for the numerically obtained results and compare the resulting image to
the experimental cell masks.
To properly achieve this, the cell lineage of the data needs to be mapped to the numerical results.
We use an in-image encoding scheme, where color values are uniquely assigned to particular cells
across the duration of the complete simulation time.
\autoref{fig:parameter-estimation} (D,E) show two images of differing configurations which are being
compared to each other.
When only comparing color values directly, we arrive at subfigure (F) where differences are
highlighted in white and matches are black.
However, since the time of the division event is not precisely determined and segmentation tools are
prone to mislabeling already divided bacteria, we also need to take into account the parental
relationship between the bacteria.
This means that upon comparing two color values which are related by either one color corresponding
to a cell that is the daughter of the other, that the assigned cost value has to be adjusted.
This procedure is shown in \autoref{fig:parameter-estimation} (G) where differences with parental
relationships between cells are indicated in gray.

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
    \end{minipage}\\
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
        Exemplary likelihood profiles for (A) stokesian damping constant, (B) strength of the
        interaction potential and (C) radius (thickness) of the bacterial rod.
        These profiles can be generated by estimating the parameters of the provided model with
        experimental microscopic images.
        Afterwards, each individual parameter is scanned within its optimization interval while the
        remaining parameters are reoptimized.
    }
    \label{fig:parameter-estimation}
\end{figure}

# Acknowledgements

# References

