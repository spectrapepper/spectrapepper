---
title: "spectrapepper: A Python toolbox for advanced analysis of spectroscopic data for materials and devices."
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
    - name: Fabien Atlan
      orcid: 0000-0002-8357-5824
      affiliation: "1"
    - name: Alejandro Perez-Rodriguez
      orcid: 0000-0002-8357-5824
      affiliation: "1, 2"
    - name: Maxim Guc
      orcid: 0000-0002-8357-5824
      affiliation: "1"
    - name: Victor Izquierdo-Roca
      orcid: 0000-0002-8357-5824
      affiliation: "1"
affiliations:
    - name: Catalonia Institute for Energy Research, Sant Adrià de Besòs, Barcelona, Spain
      index: 1
    - name: Departament d'Enginyeria Electrònica i Biomèdica, IN2UB, Universitat de Barcelona, C/ Martí i Franqués 1, 08028 Barcelona, Spain
      index: 2
date: 03 August 2021
bibliography: paper.bib
---

# Summary

Advanced material research is mainly driven by spectroscopic characterization tools, such as Raman spectroscopy,
photoluminescence, x-ray fluorescence, transmittance, reflectance, and others. These methodologies respond to the
complex structures that novel materials and devices possess and to the need for deep characterization in a
non-destructive way to be studied in-depth and better understand their properties, failure mechanisms, and possible
improvements [@Grau-Luque2021]. However, the data obtained from these methods is normally of small size (tens to a few
hundred) and subject to simple analytical procedures that are time-consuming, inefficient, and undermines the potential
of the acquired information. To improve this, methodologies that include combinatorial analysis, artificial intelligence,
and machine learning have been implemented in few studies and slowly becoming more common [@Chen2020]. Nevertheless, the
employment of such tools requires substantial amounts of high-quality data that enables the precise control of parameters
on new material synthesis, which would need the use of automated systems for characterization measurements. The
widespread utilization of such automated systems, coupled with advanced data analysis, is foreseen to shorten development
times by a factor of 10 [@Maine2006] [@Mueller2016], from 10 to 20 years [@AlanAspuru-Guzik2018] [@Correa-Baena2018] to
just a couple few. Unfortunately, several barriers to implement these tools have been identified, including the need to
have deep theoretical, statistical, analytical, programming knowledge, and the availability of tools that enable
high-data output in experiments [@Mahmood2021] [@Gu2019]. Therefore, simple and practical platforms that help researchers
to apply such tools are paramount to accelerate the universal adoption of them and ultimately shorten development times
[@Butler2018].

**spectrapepper** is a Python package that aims to ease and accelerate the use of advanced tools such as machine learning
and combinatorial analysis, through simple, straightforward, and intuitive code and functions. This library includes a
wide range of tools for spectroscopic data analysis in every step, including data acquisition, processing, analysis,
results, and visualization. Ultimately, **spectrappeper** enables the design of automated measurement systems for
spectroscopy and the combinatorial analysis of big data through high statistics, artificial intelligence, and machine
learning. **spectrapepper** is built in Python 3 [@VanRossumGuidoDrake2009], and also uses third-party packages
including numpy [@Harris2020], pandas [@Reback2021], scipy [@Virtanen2020], and matpotlib [@Caswell2021], and encourages
the user to use scikit-learn [@Pedregosa2011] for machine learning applications. **spectrapepper** comes with full
documentation, including quick start, examples, and contribution guidelines. Source code and documentation can  be
downloaded from https://github.com/enricgrau/spectrapepper.


# Features

A brief list of features include, but not limited to:

- Baseline removal functions.
- Normalization methods.
- Low and high-pass filters and trimming tools.
- Combinatorial analysis tools, such as Spearman, Pearson, and n-dimensional correlation coefficients.
- Tools for Machine Learning applications, such as data merging, randomization, and decision map.
- Sample data and examples.

# Acknowledgements


# References
