.. spectrapepper documentation master file, created by
   sphinx-quickstart on Thu May 27 11:36:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
============

**SpectraPepper** is a Python package that makes advanced analysis of spectroscopic data easy and accessible
through straightforward, simple, and intuitive code. This library contains functions for every stage of spectroscopic
methodologies, including data acquisition, pre-processing, processing, and analysis. In particular, advanced and high
statistic methods are intended to facilitate, namely combinatorial analysis and machine learning, allowing also
fast and automated traditional methods.

Features
--------
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


Functions
=========


.. automodule:: my_functions
   :members:


Examples
=========


Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
