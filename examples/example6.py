"""
This example shows basic analysis of a set of spectras.
"""

import spectrapepper as spep

# load data set
x, y = spep.load_spectras()

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

for i in y:
    plt.plot(x, i, lw=0.5, alpha=0.2, c='black')
plt.plot(x, avg, label='Average')
plt.plot(x, med, label='Median')
plt.plot(x, sdv, label='St. dev.')
plt.plot(x, typ, label='Typical')
plt.plot(x, rep, label='Representative')
plt.plot(x, mis, label='Minimum')
plt.plot(x, mas, label='Maximum')
plt.legend()
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Shift ($cm^{-1}$)')
