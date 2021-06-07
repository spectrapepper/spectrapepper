"""
This example shows how to use Scikit-learn for spectral data with Spectrapepper.
"""

#import the library
import my_functions as spep
import numpy as np

#load data
features = spep.load('data/spectras.txt', fromline=1)

#load the axis
# axis = spep.loadline('data/spectras.txt', line=1)

#load targets
targets = spep.load('data/targets.txt')
targets = np.array(targets).flatten()

#shuffle data
shuffled = spep.shuffle([features,targets], delratio=0.1)
features = shuffled[0]
targets = shuffled[1]

#target classification
classification = spep.classify(targets, glimits=[1.05,1.15], gnumber=0)
classtargets = classification[0]
labels = classification[1]


#machine learning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

lda = LinearDiscriminantAnalysis(n_components=2)
LDs = lda.fit(features, classtargets).transform(features)
df1 = pd.DataFrame(data = LDs, columns = ['D1', 'D2'])
df2 = pd.DataFrame(data = classtargets, columns =['T'])
final = pd.concat([df1, df2], axis = 1)


#visualization
import matplotlib.pyplot as plt

marker = ['o','v','s']
color = ["red","green","blue"]
for i in range(len(labels)):
    indicesToKeep = final['T'] == i
    plt.scatter(final.loc[indicesToKeep, 'D1'], final.loc[indicesToKeep, 'D2'],
                alpha=0.7, s = 50,  linewidths = 1,
                color = color[i], marker = marker[i])
plt.xlabel('D1')
plt.ylabel('D2')
plt.title('LDA')
plt.legend(labels, loc='best')
plt.show()