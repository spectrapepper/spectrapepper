"""
This example shows some of the normalization functions available.
"""

# import the library
import spectrapepper as spep

# load data
x, y = spep.load_spectras()

# normalize each spectra to its maximum value
norm1 = spep.normtomax(y)

# normalize to 10, that is, 10 will become 1.
norm2 = spep.normtovalue(y, val=10)

# normalize to the global maximum of all the data
norm3 = spep.normtoglobalmax(y)

# visualization
import matplotlib.pyplot as plt

sets = [y, norm1, norm2, norm3]
titles = ['Raw Spectras', 'Spectras with maximum normalized to 1', 'Spectras with 10 normalized to 1',
          'Spectras with global maximum normalized to 1']

for i,j in zip(sets, titles):
    for k in i:
        plt.plot(x, k)
    plt.title(j)
    plt.xlabel('Shift ($cm^{-1}$)')
    plt.ylabel('Counts (a.u.)')
    plt.show()
