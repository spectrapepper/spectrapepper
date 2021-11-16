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
      orcid: 0000-0001-7233-4892
      affiliation: "1"
    - name: Ignacio Becerril-Romero
      orcid: 0000-0002-7087-6097
      affiliation: "1"
    - name: Alejandro Perez-Rodriguez
      orcid: 0000-0002-3634-1355
      affiliation: "1, 2"
    - name: Maxim Guc
      orcid: 0000-0002-2072-9566
      affiliation: "1"
    - name: Victor Izquierdo-Roca
      orcid: 0000-0002-5502-3133
      affiliation: "1"
affiliations:
    - name: Catalonia Institute for Energy Research (IREC), Jardins de les Dones de Negre 1, 08930 Sant Adrià de Besòs, Spain
      index: 1
    - name: Departament d'Enginyeria Electrònica i Biomèdica, IN2UB, Universitat de Barcelona, C/ Martí i Franqués 1, 08028 Barcelona, Spain
      index: 2
date: 03 August 2021
bibliography: paper.bib
---

# Statement of need

In recent years, the complexity of novel high-tech materials and devices has increased considerably. This complexity
is primarily in the form of increasing numbers of components and broader ranges of applications. An example of the
latter is the last generation of thin-film solar cells, which comprise several functional micro- and nano-
layers including back contact, absorber, buffer, and transparent front contact. Most of these layers are complex
multicomponent compounds (Cu(In,Ga)Se2, Sb2Se3, CdTe, CdS, Zn(O,S), ZnO:Al, etc.) that require fine-tuning of their
physicochemical properties to ensure functionality and high peformance [@Chopra2004; @Powalla2018]. This embedded complexity means that further development of such devices requires advanced characterization and methodologies that allow
correlating the physicochemical data of the different layers (chemical composition, structural properties, defect
concentration, etc.) with the performance of the final devices in a fast, precise, and reliable way. In this regard,
non-destructive methodologies based on spectroscopic characterization techniques (Raman, photoluminescence, X-ray
fluorescence, reflectance, transmittance, etc.) have already been demonstrated to possess a high versatility and
potential for this type of analyses [@Dimitrievska2019; @Guc2017; @Oliva2016]. These spectroscopy-based methodologies
can provide deep information that encompasses the complexity of novel materials and devices in a non-destructive way,
providing a profound understanding of their properties, failure mechanisms, and possible improvements [@Grau-Luque2021].
The latest advances in the application of spectroscopic methodologies for complex materials and devices include the
implementation of combinatorial analysis (CA), artificial intelligence (AI) and machine learning (ML), that have been
already used in few studies and are slowly becoming more common [@Chen2020]. Furthermore, the widespread use of this
kind of tools in both laboratory environments and on-line/in-line monitoring of production lines is predicted to shorten
development times by a factor of 10, from 10 to 20 years to just a few
years [@Maine2006; @Mueller2016; @AlanAspuru-Guzik2018; @Correa-Baena2018]. Unfortunately, several barriers for
researchers to implement CA, AI, and ML remain [@Mahmood2021; @Gu2019]. One of them is the proper
pre-processing of spectroscopic data that allows not only to emphasize the relevant changes in the spectra, but also to
combine data obtained from different techniques and instruments. Also, the use of ML requires substantial amounts
of high-quality data for a precise analysis of the physicochemical parameters of new materials and devices, which necessitates
the use of automated systems for massive characterization measurements. In other words, the implementation of automated
high-throughput experiments and the capability to perform big-data pre-processing to enhance features of spectroscopic
data for ML, and subsequent CA, requires deep theoretical, statistical, analytical, and programming knowledge.
Therefore, simple and practical platforms that help researchers to apply such tools are paramount to accelerate their
universal adoption and ultimately shorten the development times of new materials and devices [@Butler2018].


# Overview

**`spectrapepper`** is a Python package that aims to ease and accelerate the use of advanced tools such as machine learning
and combinatorial analysis, through simple, straightforward, and intuitive code and functions. This library includes a
wide range of tools for spectroscopic data analysis in every step, including data acquisition, processing, analysis, and visualization. Ultimately, **`spectrapepper`** enables the design of automated measurement systems for
spectroscopy and the combinatorial analysis of big data through statistics, artificial intelligence, and machine
learning. **`spectrapepper`** is built in Python 3 [@VanRossumGuidoDrake2009], and also uses third-party packages
including `numpy` [@Harris2020], `pandas` [@Reback2021], `scipy` [@Virtanen2020], and `matpotlib` [@Hunter2007], and encourages
the user to use `scikit-learn` [@Pedregosa2011] for machine learning applications. **`spectrapepper`** comes with full
documentation, including quick start, examples, and contribution guidelines. Source code and documentation can  be
downloaded from https://github.com/spectrapepper/spectrapepper.


# Features

A brief and non-exhaustive list of features includes:

- Baseline removal functions.
- Normalization methods.
- Noise filters, trimming tools, and despiking methods [@Barton2019; @Whitaker2018].
- Chemometrics algorithms to find peaks, fit curves, and deconvolve spectra. 
- Combinatorial analysis tools, such as Spearman, Pearson, and n-dimensional correlation coefficients.
- Tools for ML applications, such as data merging, randomization, and decision boundaries.
- Sample data and examples.


# Acknowledgements

This work has received funding from the European Union's Horizon 2020 Research and Innovation Programme under grant agreement no. 952982 (Custom-Art project) and Fast Track to Innovation Programme under grant agreement no. 870004 (Solar-Win project). Authors from IREC belong to the SEMS (Solar Energy Materials and Systems) Consolidated Research Group of the “Generalitat de Catalunya” (ref. 2017 SGR 862) and are grateful to European Regional Development Funds (ERDF, FEDER Programa Competitivitat de Catalunya 2007–2013). MG acknowledges the financial support from Spanish Ministry of Science, Innovation and Universities within the Juan de la Cierva fellowship (IJC2018-038199-I).

# References
