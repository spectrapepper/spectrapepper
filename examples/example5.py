"""
This example shows how to use different distribution fittings on experimental
spectroscopic data. Student distribution is shown just as example, but it is
not suitable for the particular peak tested. It is also important to notice 
that the fit greatly depends on the resolution of the curve. If needed,
it is possible to first extrapolate to a greater resolution.
"""

# import libraries
import spectrapepper as spep
import matplotlib.pyplot as plt

# load data to fit
x, y = spep.load_spectras(sample=10) # load 1 single spectra from the data
y = spep.normtomax(y) # Normalize the maximum value to 1

# select peak to fit to
peak = 206 # approximate position of the peak (in cm-1) to be evaluated

# automatically fit the distributions to the peak in the data
gauss = spep.gaussfit(y=y, x=x, pos=peak, look=5)
lorentz = spep.lorentzfit(y=y, x=x, pos=peak, look=5)
student = spep.studentfit(y=y, x=x, pos=peak, look=5)
voigt = spep.voigtfit(y=y, x=x, pos=peak, look=5)

plt.plot(x, y, label='Spectra', lw=2, c='black')
plt.plot(x, gauss, lw=1, label='Gauss fit')
plt.plot(x, lorentz, lw=1, label='Lorentz fit')
plt.plot(x, student, lw=1, label='Student fit')
plt.plot(x, voigt, lw=1, label='Voigt fit')
plt.xlim(150, 260)
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Intensity (a.u.)')
plt.title('Automatic fit')
plt.legend()
plt.show()


# manually fit the distributions to the peak.
gauss = spep.gaussfit(y=y, x=x, pos=peak, sigma=4.4, manual=True)
lorentz = spep.lorentzfit(y=y, x=x, pos=peak, gamma=5, manual=True)
student = spep.studentfit(y=y, x=x, pos=peak, v=0.1, manual=True)
voigt = spep.voigtfit(y=y, x=x, pos=peak, sigma=4.4, gamma=5, manual=True)

plt.plot(x, y, label='Spectra', lw=2, c='black')
plt.plot(x, spep.normtomax(gauss), lw=1, label='Gauss fit')
plt.plot(x, spep.normtomax(lorentz), lw=1, label='Lorentz fit')
plt.plot(x, spep.normtomax(student), lw=1, label='Student fit')
plt.plot(x, spep.normtomax(voigt), lw=1, label='Voigt fit')
plt.xlim(150, 260)
plt.xlabel('Shift ($cm^{-1}$)')
plt.ylabel('Intensity (a.u.)')
plt.title('Manual fit')
plt.legend()
plt.show()


# show how the distribution changes by the change in the parameters
gauss = []
lorentz = []
student = []
voigt = []
for i in range(10):
    gauss.append(spep.gaussfit(sigma=4*(i+1), manual=True))
    lorentz.append(spep.lorentzfit(gamma=(5+i*2), manual=True))
    student.append(spep.studentfit(v=0.1*(1+1*i), manual=True))
    voigt.append(spep.voigtfit(gamma=(5+i*3), sigma=4*(i+3), manual=True))


# the stackplot fuinction is a nice tool to show the evolution of data
for i, j in zip([gauss, lorentz, student, voigt], ['Gauss', 'Lorentz', 'Student', 'Voigt']):
    spep.stackplot(i, offset=0, lw=3, figsize=(9, 9), xlabel='$x$',
                    ylabel=r'$\varphi (x)$', cmap='viridis', title=j)
 
