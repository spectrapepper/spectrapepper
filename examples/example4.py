"""
This example shows how to use Scikit-learn for spectral data with Spectrapepper.
"""

# import libraries
import my_functions as spep
import numpy as np

# load data
features = spep.load('data/spectras.txt', fromline=1)

# load targets
targets = spep.load('data/targets.txt')
targets = np.array(targets).flatten()

# shuffle data
shuffled = spep.shuffle([features, targets], delratio=0.1)
features = shuffled[0]
targets = shuffled[1]

# target classification
classification = spep.classify(targets, glimits=[1.05, 1.15], gnumber=0)
classtargets = classification[0]
labels = classification[1]

# machine learning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

lda = LinearDiscriminantAnalysis(n_components=2)
LDs = lda.fit(features, classtargets).transform(features)
df1 = pd.DataFrame(data=LDs, columns=['D1', 'D2'])
df2 = pd.DataFrame(data=classtargets, columns=['T'])
final = pd.concat([df1, df2], axis=1)
prediction = lda.predict(features)

# visualization    
spep.plot2dml(final, labels=labels, title='LDA', xax='D1', yax='D2')
