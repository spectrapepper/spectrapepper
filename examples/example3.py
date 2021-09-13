"""
This example shows how to use pearson and spearman matrices and grau plot.
"""

# import the library
import spectrapepper as spep
import numpy as np

# load data
data = np.transpose(spep.load_params())

# labels
labels = ['T', 'A1', 'A2', 'A3', 'A4', 'A5', 'S1', 'R1', 'R2', 'ETA', 'FF', 'JSC', 'ISC', 'VOC']

print(1)
# plot spearman
spep.spearman(data, labels)

print(2)
# plot pearson
spep.pearson(data, labels)

print(3)
# plot grau
spep.grau(data, labels)
