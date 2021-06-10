---
title: "spectrapepper: A Python toolbox for advanced analysis of spectroscopic data for energy materials"
tags:
    - Python
    - Spectroscopy
    - Energy materials
    - Combinatorial analysis
    - Machine learning
authors:
    - name: Enric Grau-Luque
      orcid: 0000-0002-8357-5824
      affiliation: "1"
affiliations:
    - name: Department of Geography, University of Tennessee, Knoxville, TN 37996, United States
      index: 1
date: 21 May 2020
bibliography: paper.bib
---

# Summary

Advanced energy material research is mainly driven by spectroscopic characterization tools, such as Raman spectroscopy,
photoluminescence, x-ray fluorescence, and normal reflectance. Even though these have a long history in material
research, the data obtained from these tools is mostly subject to simple analytical procedures that are time-consuming,
inefficient, and undermines the potential of these techniques. Furthermore, material development, from conception to
market, takes between 10 and 20 years, or more, with large amounts of resources 1,2. To shorten these times,
methodologies that include combinatorial analysis, artificial intelligence, and machine learning have been implemented
in few studies, slowly becoming more common3. Also, new and innovative materials are often of complex nature, thus they
require advanced methodologies to be studied in-depth and better understand their properties, failure mechanisms, and
possible improvements 4. In addition, the widespread adaptation of these methodologies is foreseen to shorten
development times by a factor of ten 5,6. Unfortunately, several barriers to implement these tools have been
identified and reported by researchers, including the need to have deep theoretical, statistical, analytical, and
programing knowledge, and the availability of tools that enable high-data output in experiments 7 8. Therefore, simple
and practical platforms that help researchers to implement such tools are paramount to accelerate the widespread
adoption of them and ultimately shorten development times 9.

**spectrapepper** is a Python package that aims to ease and accelerate the use of advanced tools such as machine learning
and combinatorial analysis, through simple and straightforward code. This library includes a wide range of tools for
spectroscopic data analysis in every step, including data acquisition, processing, analysis, results, and visualization.
**spectrapepper** is built in Python 310, and also uses third-party packages including numpy11, pandas12, scipy13, and
matpotlib14, and encourages the user to use scikit-learn15 for machine learning applications. **spectrapepper** comes with
full documentation, including quickstart, examples, and contribution guidelines. Source code and documentation can be
downloaded from https://github.com/enricgrau/spectrapepper.


# Features

A brief list of features include, but not limited to:
-   Baseline removal functions.
-   Normalization methods.
-   Low and high-pass filters and trimming tools.
-   Combinatorial analysis tools, such as Spearman, Pearson, and n-dimensional correlation coefficients.
-   Tools for Machine Learning applications, such as data merging, randomization, and decision map.
-   Easy export of data to text files to implement in visualization softwares.
-   [legend](url): a module for adding customized legends to interactive maps.


# Acknowledgements


# References
