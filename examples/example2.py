"""
This example shows simple processing of Raman spectras.
"""

# import the library
import spectrapepper as spep

# load data
x, y = spep.load_spectras()

# remove baseline
newdata = spep.alsbaseline(y)

# remove noise with moving average
newdata = spep.moveavg(newdata, 5)

# norm the sum to 1
newdata = spep.normsum(newdata)

# visualization
import matplotlib.pyplot as plt

for i in y:
    plt.plot(x, i)
plt.title('Original spectras')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in newdata:
    plt.plot(x, i)
plt.title('Processed spectras')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()
