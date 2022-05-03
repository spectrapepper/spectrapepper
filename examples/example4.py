"""
This example shows how to use Scikit-learn for spectral data with spectrapepper.
"""

# import libraries
import spectrapepper as spep

# load data
x, y = spep.load_spectras()

# load targets
targets = spep.load_targets()

# shuffle data
features, targets = spep.shuffle([y, targets], delratio=0.1)

# target classification
classtargets, labels = spep.classify(targets, glimits=[1.05, 1.15], gnumber=0)

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
