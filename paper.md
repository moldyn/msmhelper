---
title: 'msmhelper: A Python package for Markov state modeling of protein dynamics'
tags:
  - Python
  - markov state model
  - protein dynamics
  - MD
authors:
  - name: Daniel Nagel
    orcid: 0000-0002-2863-2646
    affiliation: 1
  - name: Gerhard Stock
    orcid: 0000-0002-3302-3044
    affiliation: 1
affiliations:
 - name: 'Biomolecular Dynamics, Institute of Physics,
          Albert-Ludwigs-Universität Freiburg, 79104 Freiburg, Germany'
   index: 1
date: 10 March 2023
bibliography: paper.bib

---

# Summary
Proteins function through complex conformational changes that can be difficult
to study experimentally. Molecular dynamics simulations provide high
spatiotemporal resolution, but generate massive amounts of data that require
specialized analysis. Markov state modeling is a common approach to interpret
simulations, which involves identifying biologically relevant conformations and
modeling the dynamics as memoryless transitions between them. Markov models can
complement experimental data with atomistic information, leading to a deeper
understanding of protein behavior.

In this work, we present `msmhelper`, a Python package that provides a
user-friendly and computationally efficient implementation for estimating and
validating Markov state models of protein dynamics. Given a set of metastable
conformational states, the package offers a wide range of functionalities to
improve model predictions, including dynamical correction techniques such as
the Hummer-Szabo projection formalism, dynamical coring, and Gaussian
filtering, as well as methods for predicting experimentally relevant timescales
and pathways.

# Statement of need
Markov state modeling (MSM) has emerged as an important tool for the analysis
of molecular dynamics (MD) simulations of protein dynamics [@Buchete08;
@Prinz11; @Bowman13; @Wang17]. In the general workflow of MSM, clustering into
hundreds to thousands of microstates is crucial to accurately represent the
free energy landscape and correct for non-optimal cuts between states. However,
to comprehend the underlying biological processes, it is essential to cluster
these microstates into a few macrostates that describe the dynamics as jumps
between biologically relevant metastable conformations. Despite the assumption
that microstate dynamics can be modeled by Markovian jumps, this is generally
not the case for coarse-grained macrostate dynamics due to insufficient
timescale separation between intrastate and interstate dynamics.
To address this challenge, `msmhelper` includes various dynamical correction
techniques, including a Gaussian filtering approach to include short-term
time-information in the geometric-based clustering step [@Nagel23], dynamical
coring [@Nagel19] to correct for spurious intrastate fluctuations at the
barrier, misclassified as interstate fluctuations, and the Hummer-Szabo
projection formalism [@Hummer15], enabling an optimal projection of microstate
dynamics onto the macrostate space.
Moreover, `msmhelper` provides an easy-to-use interface for common MSM
analyses, including the estimation of the transition matrix, the validation via
Chapman-Kolmogorov tests and implied timescales, and the estimation of
biological relevant pathways including their time scales, based on the concept
of `MSMPathfinder` [@Nagel20].

There are well-established Python packages, in particular PyEMMA [@pyemma] and
MSMBuilder [@msmbuilder], providing a comprehensive set of tools for the entire
MSM workflow, from feature extraction and projection of MD simulations onto
collective variables, to clustering and dynamical lumping to identify
metastable conformations, and ultimately to the estimation of Markov models and
the prediction of relevant dynamics. In contrast to them, `msmhelper` focuses
only on the estimation and analysis of Markov state models starting from
given (micro- and or macro-) state trajectories. 
Furthermore---to the best of the authors' knowledge---this is the first
publicly available Python implementation of some methodologies not implemented
in `PyEmma` and `MSMBuilder`, including dynamical coring, the Hummer-Szabo
projection formalism, and the estimation of waiting time based pathways.
Additionally, this package provides a rich command-line interface for common
analysis, including the creation of publication-ready figures of the implied
timescales, the Chapman-Kolmogorov test, different visualizations of the
waiting time distributions, and a comprehensive state representation. The
latter two techniques were suggested in @Nagel23, which uses `msmhelper` to
analyze protein dynamics.

Since Markov state modeling is usually done on local computers, it is important
to provide sufficiently fast performance. By using `numpy` [@numpy] and
just-in-time compilation via `numba` [@numba], all performance-critical methods
in `msmhelper` have been optimized. As a result, `msmhelper` can be much faster
than conventional multi-purpose programs such as `PyEmma`.
For example, adopting a 10-state trajectory with $10^5$ time steps, both the
run time of the MSM estimation (transition probability matrix) and its
validation by the well-established Chapman-Kolomogorov test are more than an
order of magnitude faster. If we compare the performance of the Monte chain
Monte Carlo (MCMC) propagation, which is commonly used to determine pathways
including their corresponding time scale distributions, `msmhelper` outperforms
`PyEMMA` by up to two orders of magnitude. More details and additional
benchmarks, including source code, can be found in the documentation.

# Example
![MSM of villin headpiece, data taken from @Nagel23. (top left) Compact
structural representation of the states, called contact representation, (top
center) implied timescales to validate the Markovianity of the model, (top
right) Chapman-Kolmogorov test for models based on different lag times compared
to the MD simulation, (bottom left) waiting time distribution of the folding
time for varying lag times compared, and (bottom center) detailed comparison of
folding time distributions to the MD simulation.\label{fig:cli}](figure.pdf)

In the following, we briefly demonstrate the capabilities of the provided
command-line interface. For this purpose, we use the publicly available micro-
and macrostate trajectories of the villin headpiece [see, @Nagel23]. It is
a well-studied fast folding protein, that allows us to test common MSM
analysis, including state characterization, MSM validation, and folding
timescale estimation. All results shown in \autoref{fig:cli} were generated
directly from the command-line interface of `msmhelper`.

# Acknowledgements
The computationally efficient implementations of `msmhelper` are made possible
by `numba` [@numba] and `numpy` [@numpy], the visualizations are based on 
`matplotlib` [@matplotlib] and `prettypyplot` [@prettypyplot], and the
command-line interface is realized with `click` [@click].

Furthermore, the authors thank Georg Diez, Sofia Sartore, Miriam Jäger, Emanuel
Dorbath, and Ahmed Ali for valuable discussions and testing the software
package.

This work has been supported by the Deutsche Forschungsgemeinschaft (DFG) via
the Research Unit FOR 5099 "Reducing complexity of nonequilibrium" (project No.
431945604). The authors acknowledge support by the High Performance and Cloud
Computing Group at the Zentrum für Datenverarbeitung of the University of
Tübingen and the Rechenzentrum of the University of Freiburg, the state of
Baden-Württemberg through bwHPC and the DFG through Grant Nos. INST 37/935-1
FUGG (RV bw16I016) and INST 39/963-1 FUGG (RV bw18A004).

# References
