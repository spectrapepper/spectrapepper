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
