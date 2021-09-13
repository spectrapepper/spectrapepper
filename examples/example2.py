"""
This example shows simple processing of Raman spectras. Do not use this procedure for formal processing.
"""

# import the library
import spectrapepper as spep

# load data
data = spep.load_spectras()

# get the axis
axis = data[0]
data = data[1:]

# remove baseline
newdata = spep.alsbaseline(data)

# norm the sum to 1
newdata = spep.normsum(newdata)

# remove noise
newdata = spep.moveavg(newdata, 5)


# visualization
import matplotlib.pyplot as plt

for i in data:
    plt.plot(axis,i)
plt.title('Original spectras')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in newdata:
    plt.plot(axis,i)
plt.title('Processed spectras')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()
