"""
This examples shows an application of a manual and self-resolved 
deconvolution with Voigt fittings of some of the peaks shown in the
spectra. 
"""
import spectrapepper as spep
import matplotlib.pyplot as plt
import numpy as np

# load data
x, y = spep.load_spectras(1)
y = spep.normtomax(y)

# some processing
y_b = spep.bspbaseline(y, x, points=[155, 243, 315, 450, 530])

# define peak positions and fitting ranges
positions = [95, 175, 205, 270, 285]
ranges = [4, 10, 10, 2, 2]

# calculate and save the fittings
fittings = []
for i in range(len(positions)):
    temp = spep.voigtfit(y_b, x, pos=positions[i], look=ranges[i])
    fittings.append(temp)

# residual and fitting convolution
residual = np.array(y_b)
convolution = [0 for _ in y]
for i in fittings:
    residual -= i
    convolution += i

# plot of all the fittings
colors = ['orange', 'purple', 'cyan', 'magenta', 'grey']
for i,j in zip(fittings, colors):
    plt.fill_between(x, i, 0, color=j, alpha=0.2)

plot all the other things
plt.plot(x, y, c='black', ls=':', label='Raw')
plt.plot(x, y_b, c='b', label='Baseline removed')
plt.plot(x, convolution, c='g', label='Convolution')
plt.plot(x, residual, c='r', ls='--', label='Residual')
plt.title('Deconvolutiuon example using Voigt fitting')
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Intensity ($a.u.$)')
plt.xlim(50, 330)
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()
