"""
This example shows how to use pearson and spearman matrices and grau plot.
"""

# import the library
import spectrapepper as spep

# load data
data = spep.load_params()

# labels
labels = ['T', 'A1', 'A2', 'A3', 'A4', 'A5', 'S1', 'R1', 'R2', 'ETA', 'FF', 'JSC', 'ISC', 'VOC']

# plot spearman
spep.spearman(data, labels)

# plot pearson
spep.pearson(data, labels)

# plot grau.
spep.grau(data, labels)
