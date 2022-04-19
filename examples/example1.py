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

for i in y:
    plt.plot(x, i)
plt.title('Raw Spectras')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in norm1:
    plt.plot(x, i)
plt.title('Spectras with maximum normalized to 1')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in norm2:
    plt.plot(x, i)
plt.title('Spectras with 10 normalized to 1')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()

for i in norm3:
    plt.plot(x, i)
plt.title('Spectras with global maximum normalized to 1')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Counts (a.u.)')
plt.show()
