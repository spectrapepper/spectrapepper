# SpectraPepper

[![image](https://img.shields.io/pypi/v/spectrapepper.svg)](https://pypi.python.org/pypi/spectrapepper) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/spectrapepper.svg)](https://anaconda.org/conda-forge/spectrapepper)

Introduction
============

**SpectraPepper** is a Python package that makes advanced analysis of spectroscopic data easy and accessible
through straightforward, simple, and intuitive code. This library contains functions for every stage of spectroscopic
methodologies, including data acquisition, pre-processing, processing, and analysis. In particular, advanced and high
statistic methods are intended to facilitate, namely combinatorial analysis and machine learning, allowing also
fast and automated traditional methods.

Features
________
The following is a short list of some of the main procedures that **SpectraPepper** package enables.

* Automatic and user-defined baseline removal.
* Several normalization methods.
* Noise filters, trimming, and other pre-processing tools.
* Combinatorial analysis tools, including Spearman, Pearson, and n-dimensional correlation coefficients.
* Tools for Machine Learning applications, such as data merging, randomization, and decision map.
* Easy export of data to text files to use visualization software, such as Origin.



Quickstart
----------

1. Install this library using ``pip``::

        pip install spectrapepper

2. Install this library using ``conda-forge``::

        conda install -c conda-forge spectrapepper

3. Test it by printing the logo!::

        import spectrapepper as spep
        spep.logo()


-   Free software: MIT license
-   Documentation: https://enricgrau.github.io/spectrapepper
    

## Features

-   TODO

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
