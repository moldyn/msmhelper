# -*- coding: utf-8 -*-
r"""# Manipulation and Preperation of Features

This submodule offers techniques for preparing and analysing the raw features&mdash;in case of MD simulations this can be contact distances, $\text{C}_\alpha$ distances, dihedral angles&mdash; or on the low-dimensional collective variables, e.g. tICA or PCA components. Here we do not provide methods for constructing the collective variables because they are implemented in many packages, e.g. in [`sklearn.decomposition`](https://scikit-learn.org/stable/modules/decomposition.html) or [`pyemma.coordinates.tica`](http://www.emma-project.org/latest/api/generated/pyemma.coordinates.tica.html#pyemma.coordinates.tica). This packages encompasses functions for determining the autocorrelation timescales. This function provide a comprehensive solution for selecting features/components.

The submodule is structured into the following submodules:

- [**timescales:**][msmhelper.features.timescales] This submodule contains a method for estimating the autocorrelation function.

"""
__all__ = [
    'estimate_acf',
    'estimate_autocorrelation_function',
]

from .timescales import estimate_acf, estimate_autocorrelation_function
