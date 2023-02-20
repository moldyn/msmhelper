# Getting Started with `msmhelper`

## Introduction
If you are interested in analyzing protein dynamics using Markov state modeling, this tutorial will provide you with a step-by-step guide to use `msmhelper`, a Python package, that is designed specifically for this purpose. The tutorial is divided into several sections, each of which covers a specific aspect of Markov state modeling, from the theoretical background to the practical implementation using this package. Whether you are a beginner or an experienced user, this tutorial will help you get up and running with Markov state modeling and enable you to explore the dynamics of complex protein systems.


## Disclaimer
Please note that the Python package `msmhelper` for Markov state modeling that we will be discussing in this tutorial provides only methods and tools for analyzing preprocessed data, in the form of a given state trajectory. The remaining workflow from the raw molecular dynamics (MD) trajectory, consisting of feature extraction, feature selection, dimensionality reduction, geometrical clustering, and dynamical lumping of states, is not provided by this package.

For feature extraction, we recommend using common packages like [MDAnalysis](https://www.mdanalysis.org/). For dimensionality reduction methods such as principal component analysis (PCA) or time-lagged independent component analysis (tICA), we recommend using [scikit-learn](https://scikit-learn.org/stable/) or [PyEMMA](http://emma-project.org/latest/). For clustering methods, scikit-learn provides many commonly used algorithms, such as $k$-means and hierarchical clustering. Finally, for dynamical coarse-graining, we recommend using packages such as [pyGPCCA](https://pygpcca.readthedocs.io/) and [MPP](https://moldyn.github.io/Clustering). For a more detailed review of existing methods, we recommend the following article ["Perspective: Identification of collective variables and metastable states of protein dynamics" by Sittel and Stock](https://doi.org/10.1063/1.5049637).

By combining the methods and tools provided by the Markov state modeling package with these other packages, you can perform a complete analysis of protein dynamics from a raw MD trajectory.

## Installation
Before you can use the Python package for Markov state modeling, you will need to install it on your machine. The package is available on both `pip` and `conda-forge`, so you can choose the installation method that suits your preferences.

To install the package using pip, open a terminal or command prompt and enter the following command:

```bash
python -m pip install msmhelper
```

where `msmhelper` is the name of the package.

If you prefer to use conda, you can install the package by entering the following command in your terminal:

```bash
conda install -c conda-forge msmhelper
```

Once you have installed the package, you are ready to head over to the tutorial!

## Sections
- [**Theoretical Background:**](theory) In this section, you will learn the basic concepts behind Markov state modeling and how it can be used to analyze protein dynamics. We will cover the key mathematical concepts, including transition matrices, equilibrium distributions, and Markov chains.

- [**Structure of `msmhelper`:**](msmhelper) This section gives a brief overview of the structure of the module, by describing the usage of every submodule itself. For more details refere to the [code reference](../reference).

- [**Estimation and Validation of an MSM:**](msm) Here, we will walk you through the process of constructing a Markov state model from clustered state trajectory of a protein dynamics simulation. We will cover data model creation, and model validation.

- [**Estimation of Waiting Times and Pathways:**](msm) In this section, we will show you how to estimate the timescales of protein conformational transitions and how to extract the most probable pathways from an MSM.

- [**HS-Projection:**](hummerszabo) Here, we will introduce you to the concept of the Hummer-Szabo (HS) projection, which allows for the optimal coarse-graining/reduction together with PCCA or MPP. We will show the dramantic improvements relying on this technique.

- [**Command Line Interface:**](cli) In this section, we will provide a short guide to the command line interface of `msmhelper`, which provides some common analysis and visualization functionality.

## Conclusion
By the end of this tutorial, you will have a good understanding of the theoretical underpinnings of Markov state modeling and how to use `msmhelper` to perform the practical analysis. You will also have learned about some advanced concepts, such as the Hummer-Szabo projection, and gained experience with the command line interface. We hope that this tutorial will provide you with the tools and knowledge you need to explore the fascinating world of protein dynamics!
