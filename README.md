<center>
    <img src="https://raw.githubusercontent.com/enricgrau/spectrapepper/main/docs/_static/spectrapepperlogo-alt.png" width="50%">
</center>

<!--
[![image](https://img.shields.io/pypi/v/spectrapepper.svg)](https://pypi.python.org/pypi/spectrapepper)
[![image](https://img.shields.io/conda/vn/conda-forge/spectrapepper.svg)](https://anaconda.org/conda-forge/spectrapepper)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://img.shields.io/lgtm/grade/python/g/enricgrau/spectrapepper.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/enricgrau/spectrapepper/context:python)
[![image](https://github.com/enricgrau/spectrapepper/workflows/docs/badge.svg)](https://enricgrau.github.io/spectrapepper)
[![codecov](https://codecov.io/gh/enricgrau/spectrapepper/branch/main/graph/badge.svg?token=IVM5BFGYHV)](https://codecov.io/gh/enricgrau/spectrapepper)
[![Downloads](https://static.pepy.tech/personalized-badge/spectrapepper?period=total&units=none&left_color=grey&right_color=blue&left_text=pypi%20downloads)](https://pepy.tech/project/spectrapepper)
[![image](https://img.shields.io/conda/dn/conda-forge/spectrapepper?color=blue&label=conda%20downloads)](https://anaconda.org/conda-forge/spectrapepper)
[![image](https://img.shields.io/badge/stackoverflow-Ask%20a%20question-brown?logo=stackoverflow&logoWidth=18&logoColor=white)](https://stackoverflow.com/questions/tagged/spectrapepper)
-->

**A Python package to simplify and accelerate analysis of spectroscopy data.**

* GitHub repo: https://github.com/enricgrau/spectrapepper
* Documentation: https://enricgrau.github.io/spectrapepper
* PyPI: https://pypi.python.org/pypi/spectrapepper
* Conda-forge: https://anaconda.org/conda-forge/spectrapepper
* Free software: MIT license

# Introduction

**Spectrapepper** is a Python package that makes advanced analysis of spectroscopic data easy and accessible
through straightforward, simple, and intuitive code. This library contains functions for every stage of spectroscopic
methodologies, including data acquisition, pre-processing, processing, and analysis. In particular, advanced and high
statistic methods are intended to facilitate, namely combinatorial analysis and machine learning, allowing also
fast and automated traditional methods.

# Features

The following is a short list of some main procedures that **SpectraPepper** package enables.

* Automatic and user-defined baseline removal.
* Several normalization methods.
* Noise filters, trimming, and other pre-processing tools.
* Cosmic Ray filters.
* Combinatorial analysis tools, including Spearman, Pearson, and n-dimensional correlation coefficients.
* Tools for Machine Learning applications, such as data merging, randomization, and decision map.
* Easy export of data to text files to use visualization software, such as Origin.

# Quickstart

1. Install this library using ``pip``::

        pip install spectrapepper

2. Install this library using ``conda-forge``::

        conda install -c conda-forge spectrapepper

3. Test it by printing the logo!::

        import spectrapepper as spep
        spep.logo()


# Credits


This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the
[giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
