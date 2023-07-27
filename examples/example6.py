"""
This example shows basic analysis of a set of spectras.
"""

import spectrapepper as spep

# load data set
x, y = spep.load_spectras()

# remove baseline
y = spep.bspbaseline(y, x, points=[155, 243, 315, 450, 530])

# Normalize the spectra to the maximum value.
y = spep.normtoratio(y, r1=[190, 220], r2=[165, 190], x=x)

# Calculate the averge spectra of the set.
avg = spep.avg(y)

# Calculate the median spectra of the set. That is, a synthetic spectra
# composed by the median value in each wavenumber.
med = spep.median(y)

# Calculate the standard deviation for each wavenumber.
sdv = spep.sdev(y)

# Obtain the typical sample of the set. That is, the spectra that is closer
# to the average.
typ = spep.typical(y)

# Look for the representative spectra. In other words, the spectra that is
# closest to the median.
rep = spep.representative(y)

# Calculate the minimum and maximum spectra. That is, the minimum and maximum
# values for each wavenumber. They are calculated together.
mis, mas = spep.minmax(y)

# visualiz the results
import matplotlib.pyplot as plt

curves = [avg, med, sdv, typ, rep, mis, mas]
titles = ['Average', 'Median', 'St. Dev.', 'Typical',
          'Representative', 'Minimum', 'Maximum']

for i in y:
    plt.plot(x, i, lw=0.5, alpha=0.2, c='black')
for i,j in zip(curves, titles):
    plt.plot(x, i, label=j)
plt.legend()
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Shift ($cm^{-1}$)')
plt.xlim(100, 600)
plt.ylim(-0.3, 0.9)
plt.show()
