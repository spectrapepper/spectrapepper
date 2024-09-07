=============
spectrapepper
=============

.. image:: https://img.shields.io/pypi/v/spectrapepper.svg
        :target: https://pypi.python.org/pypi/spectrapepper
.. image:: https://img.shields.io/conda/vn/conda-forge/spectrapepper.svg
        :target: https://anaconda.org/conda-forge/spectrapepper
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
        :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/lgtm/grade/python/g/spectrapepper/spectrapepper.svg?logo=lgtm&logoWidth=18
        :target: https://lgtm.com/projects/g/spectrapepper/spectrapepper/context:python
.. image:: https://github.com/spectrapepper/spectrapepper/workflows/docs/badge.svg
        :target: https://spectrapepper.github.io/spectrapepper
.. image:: https://codecov.io/gh/spectrapepper/spectrapepper/branch/main/graph/badge.svg?token=DC0QIwuYel
        :target: https://codecov.io/gh/spectrapepper/spectrapepper
.. image:: https://img.shields.io/conda/dn/conda-forge/spectrapepper.svg?color=blue&label=conda%20downloads
        :target: https://pepy.tech/project/spectrapepper
.. image:: https://static.pepy.tech/personalized-badge/spectrapepper?period=total&units=international_system&left_color=grey&left_text=pypi%20downloads&right_color=blue
        :target: https://pepy.tech/project/spectrapepper
.. image:: https://img.shields.io/badge/stackoverflow-Ask%20a%20question-brown
        :target: https://stackoverflow.com/questions/tagged/spectrapepper
.. image:: https://joss.theoj.org/papers/10.21105/joss.03781/status.svg
        :target: https://doi.org/10.21105/joss.03781

**A Python package to simplify and accelerate analysis of spectroscopy data.**

* GitHub repo: https://github.com/spectrapepper/spectrapepper
* Documentation: https://spectrapepper.github.io/spectrapepper
* PyPI: https://pypi.python.org/pypi/spectrapepper
* Conda-forge: https://anaconda.org/conda-forge/spectrapepper
* Free software: MIT license

Introduction
============

**spectrapepper** is a Python package that makes advanced analysis of spectroscopic data easy and accessible
through straightforward, simple, and intuitive code. This library contains functions for every stage of spectroscopic
methodologies, including data acquisition, pre-processing, processing, and analysis. In particular, advanced and high
statistic methods are intended to facilitate, namely combinatorial analysis and machine learning, allowing also
fast and automated traditional methods.

Features
--------

The following is a short list of some of the main procedures that **spectrapepper** package enables.

- Baseline removal functions.
- Normalization methods.
- Noise filters, trimming tools, and despiking methods.
- Chemometric algorithms to find peaks, fit curves, and deconvolution of spectra.
- Combinatorial analysis tools, such as Spearman, Pearson, and n-dimensional correlation coefficients.
- Tools for Machine Learning applications, such as data merging, randomization, and decision boundaries.
- Sample data and examples.

Quickstart
----------

1. Install this library using ``pip``::

        pip install spectrapepper

2. Install this library using ``conda-forge``::

        conda install -c conda-forge spectrapepper

3. Test it by plotting some data!::

        import spectrapepper as spep
        import matplotlib.pyplot as plt

        x, y = spep.load_spectras()
        for i in y:
            plt.plot(x, i)
        plt.xlabel('Raman shift ($cm^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()

4. If you find this library useful, please consider a reference or citation as::

        Grau-Luque et al., (2021). spectrapepper: A Python toolbox for advanced analysis
        of spectroscopic data for materials and devices. Journal of Open Source Software,
        6(67), 3781, https://doi.org/10.21105/joss.03781

5. Stay up-to-date by updating the library using::

        pip install --update spectrapepper
        conda update spectrapepper

6. If you encounter problems when updating, try uninstalling and then re-installing::

        pip uninstall spectrapepper
        conda remove spectrapepper

Credits
-------

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`__ and the `giswqs/pypackage <https://github.com/giswqs/pypackage>`__ project template.
