Examples
--------

Example 1: Normalization methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Thie example show the use of some of the normalization techniques included in spectrapepper::

        """
        This example shows some of the normalization functions available.
        """

        #import the library
        import my_functions as spep

        #load data
        data = spep.load('data/spectras.txt', fromline=1)

        #load the axis
        axis = spep.loadline('data/spectras.txt', line=1)

        #normalize each spectra to its maximum value
        norm1 = spep.normtomax(data)

        #normalize to a particular value (10)
        norm2 = spep.normtovalue(data,val=10)

        #normalize to the global maximum of all the data
        norm3 = spep.normtoglobalmax(data)


        #visualization
        import matplotlib.pyplot as plt

        for i in norm1:
            plt.plot(axis,i)
        plt.title('Spectras with maximum normalized to 1')
        plt.xlabel('Shift (cm-1)')
        plt.ylabel('Counts (a.u.)')
        plt.show()

        for i in norm2:
            plt.plot(axis,i)
        plt.title('Spectras with 10 normalized to 1')
        plt.xlabel('Shift (cm-1)')
        plt.ylabel('Counts (a.u.)')
        plt.show()

        for i in norm3:
            plt.plot(axis,i)
        plt.title('Spectras with global maximum normalized to 1')
        plt.xlabel('Shift (cm-1)')
        plt.ylabel('Counts (a.u.)')
        plt.show()


Examples 2: Spectral pre-processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-processing is an important step when analyzing spectral data. This examples shows how to use spectrapepper to perform
important processing tasks::

        """
        This example shows typical processing of Raman spectras.
        """

        #import the library
        import my_functions as spep

        #load data
        data = spep.load('data/spectras.txt', fromline=1)

        #load the axis
        axis = spep.loadline('data/spectras.txt', line=1)

        #norm the sum to 1
        newdata = spep.normsum(data)

        #remove noise
        newdata = spep.moveavg(newdata,5)

        #remove baseline
        newdata = spep.alsbaseline(newdata)


        #visualization
        import matplotlib.pyplot as plt

        for i in data:
            plt.plot(axis,i)
        plt.title('Original spectras')
        plt.xlabel('Shift (cm-1)')
        plt.ylabel('Counts (a.u.)')
        plt.show()

        for i in newdata:
            plt.plot(axis,i)
        plt.title('Processed spectras')
        plt.xlabel('Shift (cm-1)')
        plt.ylabel('Counts (a.u.)')
        plt.show()


Example 3: Pearson, Spearman, and Grau analyses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pearson, Spearman, and Grau plots allow to visualize possible correlations of several variables at the same time.
Spectrapepper enables these techniques with simple code, as this examples shows::

        """
        This example shows how to use pearson and spearman matrices and grau plot.
        """

        #import the library
        import my_functions as spep

        #load data
        data = spep.load('data/params.txt')

        #labels
        labels = ['T','A1','A2','A3','A4','A5','S1','R1','R2','ETA','FF','JSC','ISC','VOC']

        #plot spearman
        spep.spearman(data,labels)

        #plot pearson
        spep.pearson(data,labels)

        #plot grau
        spep.grau(data,labels)


Example 4: Machine learning preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spectrappeper includes useful functions to use along machine learning libraries, like scikit-learn::

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
