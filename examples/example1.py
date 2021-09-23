"""
This example shows some of the normalization functions available.
"""

# import the library
import spectrapepper as spep

# load data
data = spep.load_spectras()

# get the axis
axis = data[0]
data = data[1:]

# normalize each spectra to its maximum value
norm1 = spep.normtomax(data)

# normalize to a particular value (10)
norm2 = spep.normtovalue(data, val=10)

# normalize to the global maximum of all the data
norm3 = spep.normtoglobalmax(data)


# visualization
import matplotlib.pyplot as plt

for i in data:
    plt.plot(axis, i)
plt.title('Raw Spectras')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in norm1:
    plt.plot(axis, i)
plt.title('Spectras with maximum normalized to 1')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in norm2:
    plt.plot(axis, i)
plt.title('Spectras with 10 normalized to 1')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in norm3:
    plt.plot(axis, i)
plt.title('Spectras with global maximum normalized to 1')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()
