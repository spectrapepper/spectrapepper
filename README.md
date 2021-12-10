<center>
    <img src="https://raw.githubusercontent.com/spectrapepper/spectrapepper/main/docs/_static/spectrapepperlogo-alt.png" width="50%">
</center>

[![image](https://img.shields.io/pypi/v/spectrapepper.svg)](https://pypi.python.org/pypi/spectrapepper)
[![image](https://img.shields.io/conda/vn/conda-forge/spectrapepper.svg)](https://anaconda.org/conda-forge/spectrapepper)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://img.shields.io/lgtm/grade/python/g/spectrapepper/spectrapepper.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/spectrapepper/spectrapepper/context:python)
[![image](https://github.com/spectrapepper/spectrapepper/workflows/docs/badge.svg)](https://spectrapepper.github.io/spectrapepper)
[![codecov](https://codecov.io/gh/spectrapepper/spectrapepper/branch/main/graph/badge.svg?token=DC0QIwuYel)](https://codecov.io/gh/spectrapepper/spectrapepper)
[![Downloads](https://static.pepy.tech/personalized-badge/spectrapepper?period=total&units=none&left_color=grey&right_color=blue&left_text=pypi%20downloads)](https://pepy.tech/project/spectrapepper)
[![image](https://img.shields.io/conda/dn/conda-forge/spectrapepper?color=blue&label=conda%20downloads)](https://anaconda.org/conda-forge/spectrapepper)
[![image](https://img.shields.io/badge/stackoverflow-Ask%20a%20question-brown?logo=stackoverflow&logoWidth=18&logoColor=white)](https://stackoverflow.com/questions/tagged/spectrapepper)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03781/status.svg)](https://doi.org/10.21105/joss.03781)

**A Python package to simplify and accelerate analysis of spectroscopy data.**

* GitHub repo: https://github.com/spectrapepper/spectrapepper
* Documentation: https://spectrapepper.github.io/spectrapepper
* PyPI: https://pypi.python.org/pypi/spectrapepper
* Conda-forge: https://anaconda.org/conda-forge/spectrapepper
* Free software: MIT license

# Introduction

**spectrapepper** is a Python package that makes advanced analysis of spectroscopic data easy and accessible
through straightforward, simple, and intuitive code. This library contains functions for every stage of spectroscopic
methodologies, including data acquisition, pre-processing, processing, and analysis. In particular, advanced and high
statistic methods are intended to facilitate, namely combinatorial analysis and machine learning, allowing also
fast and automated traditional methods.

# Features

The following is a short list of some main procedures that **spectrapepper** package enables.

- Baseline removal functions.
- Normalization methods.
- Noise filters, trimming tools, and despiking methods.
- Chemometric algorithms to find peaks, fit curves, and deconvolution of spectra.
- Combinatorial analysis tools, such as Spearman, Pearson, and n-dimensional correlation coefficients.
- Tools for Machine Learning applications, such as data merging, randomization, and decision boundaries.
- Sample data and examples.

# Quickstart

1. Install this library using ``pip``:

        pip install spectrapepper

2. Install this library using ``conda-forge``:

        conda install -c conda-forge spectrapepper

3. Test it by plotting some data!:

        import spectrapepper as spep
        import matplotlib.pyplot as plt

        data = spep.load_spectras()
        for i in data[1:]:
            plt.plot(data[0], i)
        plt.xlabel('Raman shift ($cm^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()

# Credits


This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the
[giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
