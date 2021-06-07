"""
This main module that contains all the functions, since this package is
function-based for simplicity. Keep in mind that some of this functions are
only meant to be used inside this file, however they all might be useful in
other code. asdasd
"""

import math
import copy
import random
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import interpolate
from scipy.stats import stats
from scipy.signal import butter,filtfilt
from scipy.optimize import minimize
# import scipy.interpolate as si
from scipy.interpolate import splev, splrep
# from scipy.sparse.csgraph import _validation  
from scipy.sparse.linalg import spsolve
import itertools
from itertools import combinations
import statistics as sta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import linecache


def load_data(file):
    """
    Load data from a standard text file obtained from LabSpec and other
    spectroscopy instruments. When single measurement these come in columns with
    the first one being the x-axis. When it is a mapping, the first row is the
    x-axis and the following are the measurements. Sometimes the first 2 columns
    will be the coordinates. No headers, yet.

    :type file: str
    :param file: Url of data file. Must not have headers and separated by 'spaces' (LabSpec).

    :returns: List of the data.
    :rtype: list[float]
    """  
    new_data = []
    raw_data = open(file, "r")
    for row in raw_data: 
        row = row.replace(",", ".")
        row = row.replace(";", " ")
        row = row.replace("NaN", "-1")
        row = row.replace("nan", "-1")
        row = row.replace("--", "-1")
        s_row = str.split(row)
        s_row = np.array(s_row, dtype=np.float)
        new_data.append(s_row)
    raw_data.close()

    return new_data


def load(file, fromline = 0):
    """
    Load data from a standard text file obtained from LabSpec and other
    spectroscopy instruments. Normally, when single measurement these come in 
    columns with the first one being the x-axis. When it is a mapping, the
    first row is the x-axis and the following are the measurements.

    :type file: str
    :param file: Url of data file. Must not have headers and separated by 'spaces' (LabSpec).
    
    :type fromline: int
    :param fromline: Line of file from which to start loading data. The default is 0.
    
    :returns: List of the data.
    :rtype: list[float]
    """  
    new_data = []
    raw_data = open(file, "r")
    i = 0
    for row in raw_data: 
        if i >= fromline:
            row = row.replace(",", ".")
            row = row.replace(";", " ")
            row = row.replace("NaN", "-1")
            row = row.replace("nan", "-1")
            row = row.replace("--", "-1")
            s_row = str.split(row)
            s_row = np.array(s_row, dtype=np.float)
            new_data.append(s_row)
        i += 1        
    raw_data.close()

    return new_data


def loadheader(file, line, split=False):
    """
    Random access to file. Loads a specific header line in a file. Useful when
    managing large data files in processes where time is important. It can
    load headers as strings. If used with numbers they will be set as string.

    :type file: str
    :param file: Url od the data file

    :type line: int
    :param line: Line number. Counts from 1.    
    
    :type split: boolean
    :param split: True to make a list of strings, separated by space. The default is False.

    :returns: Array with the desierd line.
    :rtype: string
    """
    line = int(line)
    
    info = linecache.getline('raman.txt',line)
    
    if split:     
        info = str.split(info)
        
    info = np.array(info, dtype=np.str)
    
    return info


def loadline(file, line, tp='float'):
    """
    Random access to file. Loads a specific line in a file. Useful when
    managing large data files in processes where time is important. It can
    load numbers as floats.

    :type file: str
    :param file: Url od the data file

    :type line: int
    :param line: Line number. Counts from 1.     

    :returns: Array with the desierd line.
    :rtype: list[float]
    """
    line = int(line)
    file = str(file)
    
    info = linecache.getline(file,line)
    
    # print(len(info))
    # print(info[0])
    # x = info[0].isdigit()
    # print(x)
    
    info = info.replace("NaN", "-1")
    info = info.replace("nan", "-1")
    info = info.replace("--", "-1")
    
    info = str.split(info)
    
    if tp == 'float':
        info = np.array(info, dtype=np.float)
    
    if tp == 'str':
        info = np.array(info, dtype=np.str)
    
    return info

def butter_lowpass_filter(data, cutoff=0.25, fs=30, order=2, nyq=0.75):
    """
    Butter low pass filter for a single or spectra or a list of them.
        
    :type data: list[float]
    :param data: List of vectors in line format (each line is a vector).
    
    :type cutoff: float
    :param cutoff: Desired cutoff frequency of the filter. The default is 0.25.
    
    :type fs: int
    :param fs: Sample rate in Hz . The default is 30.
    
    :type order: int
    :param order: Sin wave can be approx represented as quadratic. The default is 2.
    
    :type nyq: float
    :param nyq: Nyquist frequency, 0.75*fs is a good value to start. The default is 0.75*30.

    :returns: Filtered data
    :rtype: list[float]
    """
    y = list(data) # so it does not change the input list
    normal_cutoff = cutoff / (nyq*fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # dim = np.array(y)
    if len(np.array(y).shape) > 1:
        for i in range(len(y)):
            y[i] = filtfilt(b, a, y[i])
    else:
        y = filtfilt(b, a, y)
    return y    


def normtomax(data): 
    """
    Normalizes spectras to the maximum value of each, in other words, the
    maximum value of each spectras is set to 1.

    :type data: list[float]
    :param data: Single or multiple vectors to normalize.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(data) # so it does not chamge the input list
    dims = len(np.array(y).shape) # detect dimensions
    if dims >= 2:
        for i in range(len(y)):
            max_data = max(y[i])
            for j in range(len(y[i])):
                y[i][j] = y[i][j]/max_data
    else:
        max_data = max(y)
        for i in range(len(y)):
            y[i] = y[i]/max_data
    return y


def normtovalue(data, val):
    """
    Normalizes the spectras to a set value, in other words, the define value
    will be reescaled to 1 in all the spectras.

    :type data: list[float]
    :param data: Single or multiple vectors to normalize.

    :type val: float
    :param val: Value to normalize to.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(data) # so it does not chamge the input list
    dims = len(np.array(data).shape) # detect dimensions
    if dims >= 2:
        for i in range(len(y)): # for all spectras
            for j in range(len(y[i])): # for all elements in the spectra
                y[i][j] = y[i][j]/val # divide by the maximum      
    else:
        for i in range(len(y)): # for all spectras
            y[i] = y[i]/val
    return y


def baseline_als(y, lam=100, p=0.001, niter=10):
    """
    Calculation of the baseline using Asymmetric Least Squares Smoothing. This
    script only makes the calculation but it does not remove it. Original idea of
    this algorythm by P. Eilers and H. Boelens (2005), and available details at:
    https://stackoverflow.com/a/29185844/2898619

    :type y: list[float]
    :param y: Specra to calculate the baseline from.

    :type lam: int
    :param lam: Lambda, smoothness. The default is 100.

    :type p: float
    :param p: Asymetry. The default is 0.001.

    :type niter: int
    :param niter: Niter. The default is 10.

    :returns: Returns the calculated baseline.
    :rtype: list[float]
    """    
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z) 
    return z 


def alsbaseline(data, lam=100000, p=0.001, niter=10):
    """
    This takes baseline_als and removes it from the spectra.

    :type data: list[float]
    :param data: list of spectras

    :type lam: int
    :param lam: also known as lambda, smoothness

    :type p: float
    :param p: asymetry

    :type niter: int
    :param niter: niter

    :returns: The spectra with the removed als baseline.
    :rtype: list[float]
    """    
    y = list(data)
    for i in range(len(y)):
        y[i] = y[i] - baseline_als(y[i], lam, p, niter)
    return y


def bspbaseline(data, x_axis, points, avg=5, remove=False):
    """
    Calcuates the baseline using b-spline. Find useful details and guidelines
    in https://stackoverflow.com/a/34807513/2898619.

    :type data: list[float]
    :param data: Single vector.

    :type x_axis: list[float]
    :param x_axis: x axis of the data, to interpolate the baseline function.

    :type points: list[int]
    :param n: positions in axis of points to calculate the bspine.

    :type avg: int
    :param avg: points to each side to make average.

    :type avg: bool
    :param avg: if True, calculates and returns (data - baseline).

    :returns: The baseline.
    :rtype: list[float]
    """
    data = list(data)
    x_axis = list(x_axis)
    x = list(points)
    avg = int(avg)
    
    y = [] # y values for the selected x
    for i in range(len(points)): # for all the selected points
        temp = np.mean(data[points[i] - avg: points[i] + avg + 1]) # average
        y.append(temp) # append to y   
    
    for i in range(len(x)):
        x[i] = x_axis[x[i]] # change x position to x axis value
    
    spl = splrep(x, y)
    baseline = splev(x_axis, spl)
    
    return baseline


def lorentzian_fit(data_x, ax, wid = 4, it = 100):
    """
    Fit a Lorentz distributed curve for a single spectra. Good guidelines
    for similar structures can be found at http://emilygraceripka.com/blog/16.

    :type data_x: TYPE
    :param data_x: single spectra.

    :type ax: TYPE
    :param ax: x-axis.

    :type wid: TYPE, optional
    :param wid: fitted curve width. The default is 4.

    :type it: TYPE, optional
    :param it: number of iterations. The default is 100.

    :returns: Lorentz fit.
    :rtype: list[float]
    """
    x = list(ax) # change the variable so it doesnt change the original
    y = list(data_x) # change the variable so it doesnt change the original
    cen = 0 # position (wavelength, x value)
    peak_i_pos = 0 # position (axis position, i value)
    amp = max(y) # maximum value of function, peak value, amplitud of the fit
    fit = [] # fit function
    err = [] # error log to choose the optimum
    for i in range(len(y)): # for all the data
        if amp == y[i]: # if it is the maximum
            cen = ax[i] # save the axis value x
            peak_i_pos = int(i) # also save the position i
            break  
    for i in range(it):  # search "it" iterations
        temp = 0 # reset temporal error
        fit = [] # reset lorentz fit array
        p = cen - 0.5 + ( 0.01 * i ) # pos of the peak respect to the reference measured peak
        for j in range(len(x)): # for all the points
            fit.append(amp*wid**2/((x[j]-p)**2+wid**2))
        for j in range(peak_i_pos - 25, peak_i_pos + 25): # error between -25 and +25 of the peak position
            temp = temp + ( fit[j] - y[j] ) ** 2 # total error    
        err.append(temp) # log error 
    fit = [] # reset array
    for i in range(len(err)): # look for the minimum error
        if min(err) == err[i]: # if it is the minimum error
            p = cen - 0.5 + ( 0.01 * i ) # then calculate the fit again
            for j in range(len(y)):
                fit.append(amp*wid**2/((x[j]-p)**2+wid**2))
            break
    return fit


def gaussian_fit(data_x, ax, sigma=4.4, it=100):
    """
    Fit a Gaussian distributed curve for a single spectra. Good guidelines
    for similar structures can be found at http://emilygraceripka.com/blog/16.

    :type data_x: list[float]
    :param data_x: single spectra.

    :type ax: list[float]
    :param ax: x-axis.

    :type sigma: float
    :param sigma: sigma parameter of distribution. The default is 4.4.

    :type it: int
    :param it: number of iterations. The default is 100.

    :returns: Gaussian fit.
    :rtype: list[float]
    """
    x = list(ax) # change the variable so it doesnt change the original
    y = list(data_x) # change the variable so it doesnt change the original
    cen = 0 # position (wavelength, x value)
    peak_i_pos = 0 # position (axis position, i value)
    amp = max(y) # maximum value of function, peak value,a mplitud of the fit
    fit = [] # fit function
    err = [] # error log to choose the optimum
    for i in range(len(y)): # for all the data
        if amp == y[i]: # if it is the maximum
            cen = ax[i] # save the axis value x
            peak_i_pos = int(i) # also save the position i
            break
    for i in range(it):  # search "it" iterations
        temp = 0 # reset temporal error
        fit = [] # reset lorentz fit array
        p = cen - 0.5 + ( 0.01 * i ) # pos of the peak respect to the reference measured peak
        for j in range(len(x)): # for all the points
            fit.append((11*amp)*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((x[j]-p)/sigma)**2))))
        for j in range(peak_i_pos - 25, peak_i_pos + 25): # error between -25 and +25 of the peak position
            temp = temp + ( fit[j] - y[j] ) ** 2 # total error    
        err.append(temp) # log error 
    fit = [] # reset array
    for i in range(len(err)): # look for the minimum error
        if min(err) == err[i]: # if it is the minimum error
            p = cen - 0.5 + ( 0.01 * i ) # then calculta the fit again
            for j in range(len(y)):
                fit.append((11*amp)*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((x[j]-p)/sigma)**2))))
            break
    return fit


def cortopos(vals, ax):
    """
    To translate values to a position in an axis, basically searching for
    the position of  avalue. Normally they dont fit perfectly, so it is useful
    to use this tool that aproximates to the closest.

    :type vals: list[float]
    :param vals: List of values to be searched and translated.

    :type ax: list[float]
    :param ax: Axis.

    :returns: Position in the axis of the values in vals
    :rtype: list[int]
    """
    axis = list(ax) # axis
    y = list(vals) # axis values that you want to transalte to position
    
    if len(np.array(y).shape) > 1:
        pos = [[0 for _ in range(len(y[0]))] for _ in range(len(y))] # i position of area limits
        for i in range(len(y)): # this loop takes the aprox x and takes its position
            for j in range(len(y[0])):
                dif_temp = 999 # safe initial difference
                temp_pos = 0 # temporal best position
                for k in range(len(axis)): # search in x_axis
                    if abs(y[i][j] - axis[k]) < dif_temp: # compare if better
                        temp_pos = k # save best value
                        dif_temp = abs(y[i][j] - axis[k]) # calculate new diff
                y[i][j] = axis[temp_pos] # save real value in axis
                pos[i][j] = temp_pos # save the position
    else:
        pos = [] # i position of area limits
        for i in range(len(y)): # this loop takes the aprox x and takes its position
            dif_temp = 999 # safe initial difference
            temp_pos = 0 # temporal best position
            for k in range(len(axis)):
                if abs(y[i] - axis[k]) < dif_temp:
                    temp_pos = k
                    dif_temp = abs(y[i] - axis[k])
            y[i] = axis[temp_pos] # save real value in axis
            pos.append(temp_pos) # save the position
    return pos


def areacalculator(x_data, limits, norm=False):
    """
    Area calculator using the data (x_data) and the limits in position, not
    values.

    :type x_data: list[float]
    :param x_data: Data to calculate area from

    :type limits: list[int]
    :param limits: Limits that define the areas to be calculated.
    
    :type norm: bool
    :param norm: If True, normalized the area to the sum under all the curve.

    :returns: A list of areas according to the requested limits.
    :rtype: list[float]
    """  
    data = list(x_data) # data
    lims = list(limits) # 
    
    areas = [[0 for _ in range(len(lims))] for _ in range(len(data))] # final values of areas
    for i in range(len(data)): # calculate the areas for all the points
        for j in range(len(lims)): # for all the areas
            areas[i][j] = np.sum(data[i][lims[j][0]:lims[j][1]]) # calculate the sum
            if norm:
                areas[i][j] = areas[i][j] / np.sum(data[i])
    return areas


def bincombs(n, s_min=1, s_max=0):
    """
    Returns all possible unique combinations.

    :type n: int
    :param n: Amount of positions.

    :type s_min: int
    :param s_min: Minimum amount of 1s.

    :type s_max: int
    :param s_max: Maximum amount of 1s.

    :returns: List of unique combinations.
    :rtype: list[tuple]
    """
    n = int(n)
    s_max = int(s_max)
    s_min = int(s_min)

    if s_max == 0:
        s_max = n

    e = 0 # counter
    iters = [] # final matrix
    temp = [] # temp matrix
    stuff = [] # matrix of ALL combinations

    for i in range(n): # create possibilities vector (0 and 1)
        stuff.append(1)
        stuff.append(0)
    
    for subset in itertools.combinations(stuff, n): 
        temp.append(subset) # get ALL combinations possible from "stuff"
            
    for i in range(len(temp)): # for all the possibilities...
        e = 0
        for j in range(len(iters)): # check if it already exists
            if temp[i] == iters[j]: # if it matches, add 1
                e += 1
            else:
                e += 0 # if not, add 0
        if e == 0 and sum(temp[i]) >= s_min and sum(temp[i]) <= s_max:
            iters.append(temp[i]) # if new and fall into the criteria, then add
              
    return iters 


def normsum(data):
    """
    Normalizes the sum under the curve to 1, for single or multiple spectras.

    :type data: list[float]
    :param data: Single spectra or a list of them.

    :returns: Normalized data
    :rtype: list[float]
    """   
    y = copy.deepcopy(data)
    dims = len(np.array(data).shape)
    if dims >= 2:
        for i in range(len(y)):
            s = sum(y[i])
            for j in range(len(y[i])):
                y[i][j] = y[i][j]/s
    else:    
        y = list(data)
        s = sum(y)
        for i in range(len(y)):
            y[i] = y[i]/s
    return y


def normtoglobalmax(data):
    """
    Normalizes a list of spectras to the global max.

    :type data: list[float]
    :param data: List of spectras.

    :returns: Normalized data
    :rtype: list[float]
    """ 
    y = copy.deepcopy(data)
    dims = len(np.array(data).shape)
    if dims >= 2:
        for i in range(len(y)):
            s = sum(y[i])
            for j in range(len(y[i])):
                y[i][j] = y[i][j]/s
    else:    
        y = list(data)
        s = sum(y)
        for i in range(len(y)):
            y[i] = y[i]/s
    return y


def interpolation(data, x_axis):
    """
    Returns an array with the first item being the interpolated
    data and the second item being the axis.

    :type data: list[float]
    :param data: data to interpolate

    :type x_axis: list[float]
    :param x_axis: axis of data

    :returns: Interpolated data and the new axis
    :rtype: list[list[float],list[float]]
    """
    ### LOAD DATA ###

    # master_x = []
    # temp_y = []
    # files = list(files_names)
    
    # for i in range(len(files)):
    #     data_temp = load_data(files[i])
    #     master_x.append(list(data_temp[0]))
    #     y = list(np.delete(data_temp, 0, 0)) # delete axis from 
    #     y = np.delete(y, 0, 1) # delete first column
    #     y = np.delete(y, 0, 1) # delete second (first) column 
    #     temp_y.append(y)
    
    ### END OF LOAD DATA ###

    temp_y = list(data) # data, no axis
    master_x = list(x_axis) # axis


    ### NEW AXIS ###
    new_step = 1
    new_start = -1
    new_end = 99999999
    for i in range(len(master_x)):
        if min(master_x[i]) > new_start:
            new_start = math.ceil(min(master_x[i]))
        if max(master_x[i]) < new_end:
            new_end = math.floor(max(master_x[i]))
    x_new = np.arange(new_start, new_end+new_step, new_step)
    ### END NEW AXIS ###
    
    ### MASTER DATA ###
    master_y = []
    for i in range(len(temp_y)):
        for j in range(len(temp_y[i])):
                this = interpolate.interp1d(master_x[i], temp_y[i][j])
                master_y.append(this)
    ### END MASTER DATA ###
    
    ### INTERPOLATIONS ###
    for i in range(len(master_y)):
        master_y[i] = master_y[i](x_new)
    ### END INTERPOLATIONS ###
    
    return master_y, x_new


def evalgrau(data):
    """
    This function evaluates the MSE in 3 dimensions (x,y,z) for a set of
    data vectors.

    :type data: list[float]
    :param data: A list of lists (no typo) of variables to compare.

    :returns: A list with each combination and the R2 score obtained.
    :rtype: list[float]
    """
    data = list(data)
    tup = [i for i in range(len(data))] # number list of data type sets
    combs = tuple(combinations(tup, 3)) # make all combinations        
    n_o_p = len(data[0])

    r2 = [] # R2 list
    for i in range(len(combs)): # for all the combinations
        xs = data[combs[i][0] - 1] # temporal x axis
        ys = data[combs[i][1] - 1] # temporal y axis
        zs = data[combs[i][2] - 1] # temporal z axis
        a = [] # fit function parameter
        b = [] # fit function parameter
        for j in range(n_o_p): # for all the data points
            a.append([xs[j], ys[j], 1]) # A value
            b.append(zs[j]) # b value
        b = np.matrix(b).T # transpose matrix form
        a = np.matrix(a) # matrix form
        fit = (a.T * a).I * a.T * b # evaluate fir
        rss = 0 # residual sum of squares
        tss = 0 # total sum of squares

        for j in range(n_o_p): # calculate mse for all the points
            rss += (zs[j] - (xs[j]*fit[0] + ys[j]*fit[1] + fit[2]))**2 # residual sum
            tss += (zs[j] - np.mean(zs))**2 # total error
        r2.append(round(float(1-rss/tss),2)) # R2
    merged = np.c_[combs,r2] # merge to sort
    print("yeah5")
    return merged   


def groupscores(all_targets, used_targets, predicted_targets):
    """
    Calculates the individual scores for a ML algorithm (i.e.: LDA, PCA, etc).

    :type all_targets: list[int]
    :param all_targets: List of all real targets (making sure all groups are here).

    :type used_targets: list[int]
    :param used_targets: Targets to score on.

    :type predicted_targets: list[int]
    :param predicted_targets: Prediction of used_targets.

    :returns: List of scores for each group.
    :rtype: list[float]
    """
    g_count = [0 for i in range(int(max(all_targets) + 1))] # list to count points per group
    g_scores = [0 for i in range(int(max(all_targets) + 1))] # list to store the scores
    for i in range(len(g_scores)): # for all the groups
        for j in range(len(predicted_targets)): # for all the points
            if used_targets[j] == i:
                g_count[i] += 1
                if predicted_targets[j] == used_targets[j]:
                    g_scores[i] += 1            
    for i in range(len(g_scores)):
        g_scores[i] = round( g_scores[i] / g_count[i] , 2)   
    return g_scores     


def cmscore(x_points,y_points,target):
    """
    Calculates the distance between points and the center of mass (CM) of
    clusters and sets the prediction to the closest CM. This score may be
    higher or lower than the algorithm score.

    :type x_points: list[float]
    :param x_points: Coordinates of x-axis.

    :type y_points: list[float]
    :param y_points: Coordinates of y-axis.

    :type target: list[int]
    :param target: Targets of each point.

    :returns: Score by comparing CM distances. Prediction using CM distances. X-axis coords of ths CMs. Y-axis coords of the Cms.
    :rtype: list[float, list[int],list[float],list[float]]
    """
    x_p = list(x_points)
    y_p = list(y_points)
    tar = list(target)
    g_n = int(max(target) + 1)
        
    a = [0 for i in range(g_n)] # avg D1
    b = [0 for i in range(g_n)] # avg D2
    c = [0 for i in range(g_n)] # N for each group
    d = [] # distances
    p = [] # predictions
    
    for i in range(len(tar)):
        for j in range(g_n):
            if tar[i] == j:
                a[j] += x_p[i]  
                b[j] += y_p[i]
                c[j] += 1
                
    for i in range(g_n):
        a[i] = a[i]/c[i]
        b[i] = b[i]/c[i]
    
    correct = 0
    for i in range(len(tar)):
        temp1 = -1
        temp2 = 1000
        temp3 = []
        
        for j in range(g_n):
            temp3.append(((x_p[i]-a[j])**2 
            +(y_p[i]-b[j])**2)**0.5)
            
            if temp3[j] < temp2:
                temp2 = temp3[j]
                temp1 = j
        
        p.append(temp1)
        d.append(temp3) 
        
        if tar[i] == temp1:
            correct += 1 
    
    score = round(correct/len(tar),2)
    
    return score,p,a,b


def mdscore(x_p, y_p, tar):
    """
    Calculates the distance between points and the median center (MD) of
    clusters and sets the prediction to the closest MD. This score may be
    higher or lower than the algorithm score.

    :type x_p: list[float]
    :param x_p: Coordinates of x-axis.

    :type y_p: list[float]
    :param y_p: Coordinates of y-axis.

    :type tar: list[int]
    :param tar: Targets of each point.

    :returns: Score by comparing MD distances. Prediction using MD distances. X-axis coords of ths CMs. Y-axis coords of the Cms.
    :rtype: list[float, list[int],list[float],list[float]]
    """
    x_p = list(x_p)
    y_p = list(y_p)
    g_n = max(tar) + 1
    tar = list(tar)
        
    a = [0 for i in range(g_n)]
    b = [0 for i in range(g_n)]
    d = [] # distances
    p = [] # predictions
    
    x_s = [[] for _ in range(g_n)]
    y_s = [[] for _ in range(g_n)]
    
    for i in range(g_n):
        for j in range(len(tar)):
            if i == tar[j]:
                x_s[i].append(x_p[j])
                y_s[i].append(y_p[j])

    for i in range(g_n):
        a[i] = np.median(x_s[i])  
        b[i] = np.median(y_s[i])

    correct = 0
    for i in range(len(tar)):
        temp1 = -1
        temp2 = 1000
        temp3 = []
        for j in range(g_n):
            temp3.append(((x_p[i]-a[j])**2 
            +(y_p[i]-b[j])**2)**0.5)
            
            if temp3[j] < temp2:
                temp2 = temp3[j]
                temp1 = j
        
        p.append(temp1)
        d.append(temp3) 
        
        if tar[i] == temp1:
            correct += 1 
    
    score = round(correct/len(tar),2)
    
    return score,p,a,b


def normtopeak(data, x_axis, peak, shift=10):
    """
    Normalizes the spectras to a particular peak.

    :type data: list[float]
    :param data: Data to be normalized, no x-axis

    :type x_axis: list[float]
    :param x_axis: X-axis of the data

    :type peak: int
    :param peak: Peak position in x-axis values.

    :type shift: int
    :param shift: Range to look for the real peak. The default is 10.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(data)
    x_axis = list(x_axis)
    peak = [int(peak)]
    shift = int(shift)
    
    pos = cortopos(peak, x_axis)
    section = y[pos[0]-shift:pos[0]+shift]   
    highest = peakfinder(section,l=int(shift/2))
    
    c = 0
    for i in range(len(highest)):
        if highest[i] == 1:
            c = i
            break    
    
    local_max = y[pos[0]-shift + c]
    y = y / local_max
    return y
    

def peakfinder(data, l=10):
    """
    Find the location of the peaks in a single vector.

    :type data: list[float]
    :param data: Data to find a peak in.

    :type l: int
    :param l: Amount of position to each side to decide if it is a local maximum. The default is 10.

    :returns: A vector of same length with 1 or 0 if it finds or not a peak in that position
    :rtype: list[int]
    """
    y = list(data)
    look = l #positions to each side to look max or min (2*look)
    is_max = [0 for i in data]
    is_min = [0 for i in data]
    for i in range(look, len(y) - look): #start at "look" to avoid o.o.r
        lower = 0 #negative if lower, positive if higher
        higher = 0
        for j in range(look):
            if (y[i] <= y[i - look + j] and
                y[i] <= y[i + j]): #search all range lower
                lower += 1 #+1 if lower
            elif (y[i] >= y[i - look + j] and
                  y[i] >= y[i + j]): #search all range higher
                higher += 1 #+1 if higher    
        if higher == look: #if all higher then its local max
            is_max[i] = 1
            is_min[i] = 0
        elif lower == look: #if all lower then its local min 
            is_max[i] = 0
            is_min[i] = 1
        else:
            is_max[i] = 0
            is_min[i] = 0
    return is_max


def confusionmatrix(tt, tp, gn, plot=False, groupnames=['','','',''], title='',
                    colormap='Blues', fontsize=20):
    """
    Calculates and/or plots the confusion matrix for machine learning algorithm results.

    :type tt: list[float]
    :param tt: Real targets.

    :type tp: list[float]
    :param tp: Predicted targets.

    :type gn: int
    :param gn: Number of classification groups.

    :type fontsize: int
    :param fontsize: Font size for the labels. The default is 20.

    :returns: The confusion matrix
    :rtype: list[float]
    """
    tt = list(tt) # real targets
    tp = list(tp) # predicted targets
    gn = int(gn) # group number    
    plot = bool(plot)
    group_names = list(groupnames)
    title=str(title)
    colormap = str(colormap)
    fontsize = int(fontsize)
    
    m = [[0 for i in range(gn)] for j in range(gn)]
    p = [0 for i in range(gn)]
    
    for i in range(len(p)):
        for j in tt:
            if i == j:
                p[i] += 1
    for i in range(len(tt)):
        for row in range(len(p)):
            for col in range(len(p)):
                if tp[i] == col and tt[i] == row:
                    m[row][col] += 1
            
    for i in range(len(m)):
        m[i] = np.array(m[i]) / p[i]
        
    if plot:
        fig = plt.figure(tight_layout=True,figsize=(6, 7.5))
        plt.rc('font', size=fontsize)
        ax = fig.add_subplot()
        im = ax.imshow(m, cmap=colormap)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(group_names)))
        ax.set_yticks(np.arange(len(group_names)))
        ax.set_xticklabels(group_names)
        ax.set_yticklabels(group_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        # to show values in pixels
        # for i in range(len(group_names)):
            # for j in range(len(group_names)):
                # text = ax.text(j, i, round(m[i][j],2), ha='center', va='center', color='black')

    return m


def avg(data):
    """
    Calculates the average vector from a list of vectors.

    :type data: list[(]float]
    :param data: List of vectors.

    :returns: The average of the vectors in the list.
    :rtype: list[float]
    """
    data = list(data)
    avg_data = np.array([0 for i in range(len(data[0]))])
    for i in data:
        avg_data = avg_data + i
    avg_data = avg_data / len(data)
    return avg_data 


def sdev(data):
    """
    Calculates the standard deviation for each bin from a list of vectors.

    :type data: list[(]float]
    :param data: List of vectors.

    :returns: Standard deviation curve
    :rtype: list[(]float]
    """
    curve_std = [] # stdev for each step    
    data = list(data)
    for i in range(len(data[0])):
        temp = []
        for j in range(len(data)):
            temp.append(data[j][i])
        curve_std.append(sta.stdev(temp)) #stdev for each step   
    curve_std = np.array(curve_std)
    return curve_std


def peakfit(data, ax, pos, look=20, shift=5, wid=5.0):
    """
    Feats peak as an optimization problem.

    :type data: list[float]
    :param data: Data to fit. Single vector.

    :type ax: list[float]
    :param ax: x axis.

    :type pos: int
    :param pos: Peak position to fit to.

    :type look: int
    :param look: positions to look to each side. The default is 20.

    :type shift: int
    :param shift: Possible shift of the peak. The default is 5.

    :type wid: float
    :param wid: Initial width of fit. The default is 5.

    :returns: Fitted curve.
    :rtype: list[float]
    """
    # initial guesses
    ax = list(ax)
    y = list(data)
    look = int(look)
    s = int(shift)
    pos = int(pos)
    amp = max(y)/2
    wid = float(wid)
    
    def objective(x): 
        fit = []
        error = 0
        for j in range(len(ax)): # for all the points
            fit.append(x[0]*x[1]**2/((ax[j]-ax[pos])**2+x[1]**2))
        for j in range(pos - look, pos + look): # error between +-look of the peak position
            error += (fit[j]-y[j])**2 # total error                 
        return error
        
    def constraint1(x):
       return 0
    
    for i in range(pos-s,pos+s):
        if y[i] > y[pos]:
            pos = i
    
    x0 = np.array([amp,wid]) # master vector to optimize, initial values
    bnds = [[0,max(y)],[1,1000]]
    con1 = {'type': 'ineq', 'fun': constraint1}
    cons = ([con1])
    solution = minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons)
    x = solution.x
    
    fit = []
    for j in range(len(ax)): # for all the points
            fit.append(x[0]*x[1]**2/((ax[j]-ax[int(pos)])**2+x[1]**2))
    return fit        


def decbound(xx,yy,xlims,ylims,divs=0.01):
    """
    Calculates the Decision Boundaries.

    :type xx: list[float]
    :param xx: X coordinates of centroids.

    :type yy: list[float]
    :param yy: Y coordinates of centroids.

    :type xlims: list[float]
    :param xlims: Limits.

    :type ylims: list[float]
    :param ylims: Limits.

    :type divs: float
    :param divs: Resolution. The default is 0.01.

    :returns: The decision boundaries.
    :rtype: list[float]
    """
    cmx = list(xx) # centroids
    cmy = list(yy)
    divs = float(divs) # step
    map_x = list(np.array(xlims)/divs) # mapping limits
    map_y = list(np.array(ylims)/divs)
        
    x_divs = int((map_x[1]-map_x[0])) # mapping combining 'divs' & 'lims'
    y_divs = int((map_y[1]-map_y[0]))

    x_cords = [divs*i for i in range(int(min(map_x)),int(max(map_x)))] # coordinates
    y_cords = [divs*i for i in range(int(min(map_y)),int(max(map_y)))]
       
    pmap = [[0 for i in range(x_divs)] for j in range(y_divs)] # matrix
    
    for i in range(len(x_cords)):
        for j in range(len(y_cords)):
            t1 = 9999999999999999999999999
            #grad = [] #####
            for k in range(len(cmx)):
                d = (cmx[k]-x_cords[i])**2 + (cmy[k]-y_cords[j])**2
                #grad.append(d) ####
                if d < t1:
                    t1 = d
                    pmap[j][i] = k
            #grad = np.array(grad)/sum(grad) #####
            #pmap[j][i] = 0 ####
            #for k in range(len(grad)): ####
            #    pmap[j][i] += len(grad) - (k+1)*grad[k] ###
    return pmap


def regression(target, variable, cov=0, exp=0):
    """
    Performs an N dimensional regression.

    :type target: list[float]
    :param target: Y-axis values, values to predict.

    :type variable: list[float]
    :param variable: X-axis values.

    :type cov: int
    :param cov: If 1 is regression with covariance, like spearman. The default is 0.

    :type exp: int
    :param exp: For exponent (non linear) regressions. Not working yet. The default is 0.

    :returns: Prediction of the fitting and the Fitting parameters.
    :rtype: list[list[float],list[float]]
    """
    target = list(target)
    master = list(variable)
    cov = int(cov)
    # exp = int(exp) # for non-linear fittings in the future
    
    if cov == 1:
        master.append(target) # array of arrays
        print(len(master))
        pos = [i for i in range(len(master[0]))] # ascending values
        df = pd.DataFrame(data=master[0], columns=['0']) # 1st col is 1st var
        for i in range(len(master)): # for all variables and target
            df[str(2*i)] = master[i] # insert into dataframe
            df = df.sort_values(by=[str(2*i)]) # sort ascending
            df[str(1+2*i)] = pos # translate to position
            df = df.sort_index() # reorder to maintain original position
    
        master = [df[str(2*i+1)] for i in range(int(len(df.columns)/2-1))]
        target = df[str(len(df.columns)-1)]
        
    A = [] # fit function parameter
    for i in range(len(master[0])): # for all the data points
        v = [1 for i in range(len(master)+1)]
        for j in range(len(master)):
            v[j] = master[j][i]
        A.append(v) # A value
    
    b = np.matrix(target).T # transpose matrix
    A = np.matrix(A)
    fit = (A.T * A).I * A.T * b # evaluate fir

    prediction = []
    for i in range(len(master[0])):
        p = 0
        for j in range(len(master)):
            p += master[j][i]*fit[j]
        p += fit[len(fit)-1]
        prediction.append(float(p))

    return prediction,fit


def decdensity(lims, x_points, y_points, groups, divs=0.5):
    """
    Calculates the density decision map from a cluster mapping.

    :type lims: list[float]
    :param lims: Plot and calculation limits.

    :type x_points: list[float]
    :param x_points: X coordinates of each point.

    :type y_points: list[float]
    :param y_points: Y coordinates of each point.

    :type groups: list[int]
    :param groups: List of targets for each point.

    :type divs: float
    :param divs: Resolution ti calculate density. The default is 0.5.

    :returns: The density decision map.
    :rtype: list[float]
    """
    divs = float(divs) # resolution, step
    
    map_x = list(np.array(lims[:2])/divs) # mapping limits
    map_y = list(np.array(lims[2:])/divs)
    
    x_p = list(x_points) # points coordinates
    y_p = list(y_points) 
    
    groups = list(groups) # target (group) list
    n_groups = int(max(groups)+1) # number of groups
        
    x_divs = int((map_x[1]-map_x[0])) # divisons for mapping
    y_divs = int((map_y[1]-map_y[0]))
    
    x_cords = [divs*i for i in range(int(min(map_x)),int(max(map_x)))] # coordinates
    y_cords = [divs*i for i in range(int(min(map_y)),int(max(map_y)))]
               
    master = [] # to store the maps for each group
    
    for l in range(n_groups):
        pmap = [[0 for i in range(x_divs)] for j in range(y_divs)] # individual matrices
        for i in range(len(x_cords)-1):
            for j in range(len(y_cords)-1):
                count = 0
                for k in range(len(x_p)):
                    if (x_cords[i] < x_p[k] and x_p[k] < x_cords[i+1] and
                        y_cords[j] < y_p[k] and y_p[k] < y_cords[j+1] and
                        groups[k] == l):
                        count += 1
                pmap[j][i] = count  
    
        maximum = max(np.array(pmap).flatten())       
        pmap = np.array(pmap)/maximum
        master.append(pmap) 
    return master        


def colormap(c,name='my_name',n=100):
    """
    Just to simplify a bit the creation of colormaps.

    :type c: list
    :param c: List of colors thaat you ant on the colormap

    :type name: string, optional
    :param name: Name of the colormap. The default is 'my_name'.

    :type n: TYPE, optional
    :param n: Divisions. The default is 100.

    :returns: Colormap in Matplotlib format.
    :rtype: cmap
    """
    colors = list(c)  # R -> G -> B
    cmap_name = str(name)
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    return cmap


def isaxis(data):
    """
    Detects if there is an axis in the data.

    :type data: list[float]
    :param data: Data containing spectras an possible axis.

    :returns: True if there is axis.
    :rtype: bool
    """
    features = list(data)
    
    is_axis = True # there is axis by default
    x_axis = features[0] # axis should be the first
    for i in range(len(x_axis)-1): # check all the vector
        if x_axis[i] > x_axis[i+1]:
            is_axis = False
            break  
    return is_axis


def trim(data,start = 0,finish = 0):
    """
    Deletes columns in a list from start to finish.

    :type data: list
    :param data: Data to be trimmed. Single vector at the moment.

    :type start: int, optional
    :param start: Poistion of the starting point. The default is 0.

    :type finish: int, optional
    :param finish: Position of the ending point. The default is 0.

    :returns: Trimmed data.
    :rtype: list[]
    """
    data = list(data)
    s = int(start)
    f = int(finish)
    
    if f == 0 or f > len(data):
        f = len(data)
    
    t = f-s
    
    for i in range(t):
        data = np.delete(data, s, 1)
    return data


def shuffle(arrays, delratio=0):
    """
    Merges and shuffles data and then separates it so it is shuffles together.

    :type arrays: list[list[float]]
    :param arrays: List of arrays of data.

    :type delratio: float
    :param delratio: Ratio of the data to be deleted, 0 < delratio < 1

    :returns: List of the shuffled arrays.
    :rtype: list[list[float]]
    """
    all_list = list(arrays)
    delratio = float(delratio)
    
    features = all_list[0]
    for i in range(1,len(all_list)):
        features = np.c_[features,all_list[i]]

    np.random.shuffle(features) # shuffle data before training the ML
 
    ###
    if delratio > 0: # to random delete an amount of data
        if delratio >= 1:
            delratio = 0.99
            
        delnum = int(math.floor(delratio*len(features))) # amount to delete
        
        for i in range(delnum):
            row = random.randrange(0,len(features)) # row to delete
            features = np.delete(features,row , 0)  
    ###
    
    new_list = [[] for _ in all_list]
    lengths = []

    for i in range(len(all_list)):
        if len(np.array(all_list[i]).shape) >= 2: # are lists are 2d
            lengths.append(np.array(all_list[i]).shape[1]) # save the length
        else:
            lengths.append(1) # otherwise is only 1 number
    
    for i in range(len(all_list)):
        for j in range(len(features)): #all_list[0]
            if i == 0:
                new_list[i].append(features[j][0:lengths[i]])
            else:
                new_list[i].append(features[j][sum(lengths[0:i])])

    return new_list


def mergedata(data):
    """
    Merges data, it can merge large vectors. Useful to merge features
    before performing ML algorithms.

    :type data: list[list[float]]
    :param data: List of arrays.

    :returns: List with the merged data.
    :rtype: list[float]
    """
    data = list(data) # list of lists
    master = [[] for _ in data[0]]
    for i in range(len(data[0])):
        for j in range(len(data)):
            master[i].extend(data[j][i])
    return master        


def logo(lay=90, leng=100, a=1, b=0.8, r1=80, r2=120, lw=2):
    """
    Prints the logo of SpectraPepper.

    :type lay: int
    :param lay: Amount of layers/vectors. The default is 90.

    :type leng: int
    :param leng: Length of the layers. The default is 100.

    :type a: int
    :param a: Amplitud. The default is 1.

    :type b: int
    :param b: Width. The default is 0.8.

    :type r1: int
    :param r1: Random lower limit. The default is 80.

    :type r2: int
    :param r2: Random upper limit. The default is 120.

    :type lw: int
    :param lw: Line width. The default is 2.
    """
    x = [[1 for i in range(leng)] for j in range(lay)]
    
    def xl(amp,wid,p):
        return [(amp*wid**2/((i-p)**2+wid**2))*random.randint(99,101)/100 for i in range(100)]
    
    for i in range(len(x)):
        wid = 0.0000002*(i**4)-0.0002*(i**3)+0.0178*(i**2)-0.0631*i+4.7259
        pos = 0.000001*(i**4)-0.0004*(i**3)+0.046*(i**2)-1.8261*i+59.41
        amp = wid*(a)
        x[i] = xl(amp,wid*b,pos)
    
    axis = [i for i in range(leng)]
    
    plt.figure(figsize=(10, 14))
    for i in range(len(x)):
        r = [random.randint(r1,r2)/100 for i in range(leng)]
        curve = np.array(x[i])+i+r
        plt.plot(curve, color='black', lw=lw)
        plt.fill_between(axis,curve, color='white', alpha=1)
    plt.show()


def shiftref(data, peak_ref=520, mode=1, it=100, plot=True):
    """
    Shifts the x-axis according to a shift calculated prior.

    :type data: list[float]
    :param data: Reference measurement (Si), with axis.

    :type peak_ref: float
    :param peak_ref: Where the reference peak should be in x-axis values. The default is 520 (Raman Si).

    :type mode: int
    :param mode: Fitting method, Lorentz, Gaussian, or none (1,2,3). The default is 1.

    :type it: int
    :param it: Fitting iterations. The default is 100.

    :type plot: bool
    :param plot: If True plots a visual aid. The default is True.

    :returns: Shift amount
    :rtype: float
    """
    si_data = list(data)
    peak_ref = float(peak_ref)
    mode = int(mode)
    it = int(it)
    plot = bool(plot)
    
    data_trans = [] # transition array
    data_c = [] # curve (y values)
    x_axis = [] # si x axis, later it will change 
    fit = [] # fit curves(s), if selected
    shift = [] # axis shift array
    
    for i in range(len(si_data)): # conditionning data
        data_trans.append(np.transpose(si_data[i])) # transpose
        x_axis.append(data_trans[i][0]) # separate axis
        data_c.append(data_trans[i][1]) # separate curve values
        
    for i in range(len(data_c)): # depending on the mode chosen...    
        if mode == 1:
            fit.append(lorentzian_fit(data_c[i], x_axis[i], 4, it)) # check my_functions for parameters
        if mode == 2:
            fit.append(gaussian_fit(data_c[i], x_axis[i], 4.4, it))
        if mode == 0:
            fit.append(data_c[i])
    
    for i in range(len(fit)): # look for the shift with max value (peak)
        for j in range(len(fit[0])): # loop in all axis
            if fit[i][j] == max(fit[i]): # if it is the max value,
                shift.append(x_axis[i][j] - peak_ref) # calculate the diference
    
    temp = 0 # temporal variable
    for i in range(len(shift)):
        temp = temp + shift[i] # make the average
    
    peakshift = -temp / len(shift)
    
    if plot:
        plt.figure() #figsize = (16,9)
        for i in range(len(data_c)):
            plt.plot(x_axis[i], data_c[i], linewidth = 2, label='Original' + str(i))
            plt.plot(x_axis[i], fit[i], linewidth = 2, label='Fit' + str(i), linestyle='--')
        plt.axvline(x=peak_ref, ymin=0, ymax=max(data_c[0]), linewidth = 2, color = "red", label=peak_ref)
        plt.axvline(x=peak_ref - peakshift, ymin=0, ymax=max(data_c[0]), linewidth = 2, color = "yellow", label="Meas. Max.")
        plt.gca().set_xlim(peak_ref - 15, peak_ref + 15)
        #plt.gca().set_ylim(0, 2)
        plt.legend(loc=0)
        plt.ylabel('a.u.')
        plt.xlabel('Shift (cm-1)')
        plt.show()
        
    return peakshift    


def classify(data, gnumber=3, glimits=[]):
    """
    Classifies targets according to either defined limits or number of groups.
    The latter depends on the defined parameters.

    :type data: list[float]
    :param data: Vector with target values to be classified.

    :type gnumber: int
    :param gnumber: Number of groups to create. The default is 3 and is the default technique.

    :type glimits: list[float]
    :param glimits: Defined group limits. The default is [].

    :returns: Vector with the classification from 0 to N. A list with strings with the name of the groups, useful for plotting.
    :rtype: list[list,list]
    """
    group_number = int(gnumber)
    group_limits = list(glimits)
    targets = list(data)
    
    # df_targets['T'] is data from the file, ['NT'] the classification (0 to ...)
    df_targets = pd.DataFrame(data = targets, columns =['T'])
    group_names = []
    class_targets = [] # ['NT']
    
    # if I set the number of groups
    if group_number > 0:
        group_limits = [[0,0] for i in range(group_number)]
        df_targets.sort_values(by='T', inplace=True)
        group_size = math.floor(len(targets)/group_number)
        left_over = len(targets) - group_size * group_number 
        g_s = []
        for i in range(group_number):
            g_s.append(group_size + math.floor((i+1)/(group_number))*left_over)
        for i in range(len(g_s)):
            for j in range(g_s[i]):
                class_targets.append(i)    
        df_targets['NT'] = class_targets
        temp = 0
        for i in range(group_number):    
            if i == 0:
                group_limits[i][0] = df_targets['T'].iloc[0] 
            else:
                group_limits[i][0] = df_targets['T'].iloc[temp]
            if i == group_number - 1:    
                group_limits[i][1] = df_targets['T'].iloc[df_targets['T'].size - 1]
                group_names.append(str(group_limits[i][0]) + ' < ')
            else:
                group_limits[i][1] = df_targets['T'].iloc[int(temp + g_s[i])]
                group_names.append(str(group_limits[i][0]) + ' - ' + str(group_limits[i][1]))
            temp = temp + g_s[i] 
        df_targets.sort_index(inplace=True)         
    
    # if I set the limits
    if len(group_limits) >= 1 and group_number <= 1:
        class_targets = [-1 for i in range(len(targets))]
        
        for i in range(0,len(group_limits)):
            for j in range(len(targets)):
                
                if targets[j] < group_limits[0]:
                    class_targets[j] = 0
                    
                if targets[j] >= group_limits[len(group_limits)-1]:
                    class_targets[j] = len(group_limits)
                
                elif targets[j] >= group_limits[i] and targets[j] < group_limits[i + 1]:
                    class_targets[j] = i + 1
                    
        group_names.append(' < ' + str(group_limits[0]))
        for i in range(0,len(group_limits) - 1):
            group_names.append(str(group_limits[i]) + ' - ' + str(group_limits[i + 1])) 
        group_names.append(str(max(group_limits)) + ' =< ')                  
        
    return class_targets,group_names


def subtractref(data, ref, alpha=0.9, sample=0, plot=True, plot_lim=[50, 200], mcr=False):
    """
    Subtracts a reference spectra (i.e.: air) from the measurements.

    :type data: list[float]
    :param data: List of spectras, with x-axis.

    :type ref: list[float]
    :param ref: air (reference) data, with x-axis.

    :type alpha: float
    :param alpha: Manual multiplier. The default is 0.9.

    :type sample: int
    :param sample: Sample spectra to work with. The default is 0.

    :type plot: bool
    :param plot: To plot or not a visual aid. The default is True.

    :type plot_lim: list[int]
    :param plot_lim: Limits of the plot.

    :type mcr: bool
    :param mcr: Use MCR (self calculation), not available yet. The default is False.

    :returns: Data with the subtracted reference.
    :rtype: list`[float]
    """
    data = list(data)
    air = list(ref)
    sample = int(sample) # spectrum chosen to work with. 0 is the first
    alpha = float(alpha) # multiplication factor to delete the air spectrum on the sample
    plot_lim = list(plot_lim) # range of the plot you want to see

    x_axis = data[0] # x axis is first row
    data = np.delete(data, 0, 0) # delete x_axis from data
    data = list(data)
    
    if len(air[0]) <= 2: # if not in correct format
        air = np.transpose(air) # transpose so it is in rows
    
    x_axis_air = air[0] # x axis is first row
    air = air[1] # make it the same structure as x_axis
    final = data[sample] - air * alpha # final result
    
    if plot:
        plt.plot(x_axis, data[sample], linewidth = 1, label='Original', linestyle='--')
        plt.plot(x_axis_air, air, linewidth = 1, label='Air', linestyle='--')
        plt.plot(x_axis, final, linewidth = 1, label='Final')
        plt.gca().set_xlim(plot_lim[0], plot_lim[1])
        plt.legend(loc=0)
        plt.ylabel('a.u.')
        plt.xlabel('Shift (cm-1)')
        plt.show() 
    
    for i in range(len(data)):
        data[i] = data[i] - air * alpha
    
    data.insert(0,x_axis)
    
    return data


def pearson(data, labels, cm="seismic", fons=20, figs=(20, 17), tfs=25, ti="Pearson"):
    """
    Calculates Pearson matrix and plots it.

    :type data: list[float]
    :param data: Data to correlate.

    :type labels: list[str]
    :param labels: Labels of the data.

    :type cm: str
    :param cm: Color map as for matplolib. The default is "seismic".

    :type fons: int
    :param fons: Plot font size. The default is 20.

    :type figs: tuple
    :param figs: Plot size. The default is (20,17).

    :type tfs: int
    :param tfs: Title font size. The default is 25.

    :type ti: str
    :param ti: Plot title/name. The default is "spearman".
    """
    data = list(np.transpose(data))
    labels = list(labels)
    cm = str(cm)
    fonsize = float(fons)
    figsize = list(figs)
    titlefs = float(tfs)
    title = str(ti)
    n = len(data)
    
    pears = []
    cordsi = [] # coordinates, same order as labels
    cordsj = [] # coordinates, same order as labels
    for i in range(n): # for all sets
        for j in range(n):# compare with all sets
            x = data[i]
            y = data[j]
            pears.append(stats.pearsonr(x,y)[0])
            
            cordsi.append(int(i))
            cordsj.append(int(j))
    
    merged_pears = np.c_[ pears, cordsi, cordsj ] # merge to sort together 
    merged_pears = sorted(merged_pears,key=lambda l:l[0], reverse=True)     
    
    for i in range(n): # delete the first n (the obvious 1s)
        merged_pears = np.delete(merged_pears, 0, 0)
    
    for i in range(int((n*n-n)/2)): # delete the repeated half
        merged_pears = np.delete(merged_pears, i, 0)
    
    pears = np.reshape(pears,(n,n)) #[pearson coeff, p-value]
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])
    plt.rc('font', size=fonsize)
    fig = plt.figure(tight_layout=True,figsize=figsize)
    
    y = [i+0.5 for i  in range(n)]
    ticks = mpl.ticker.FixedLocator(y)
    formatt = mpl.ticker.FixedFormatter(labels)
    
    ax = fig.add_subplot(gs[0, 0])
    pcm = ax.pcolormesh(pears, cmap=cm, vmin=-1, vmax=1)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(title, fontsize=titlefs)
    ax.xaxis.set_major_locator(ticks)
    ax.yaxis.set_major_locator(ticks)
    ax.xaxis.set_major_formatter(formatt)
    ax.yaxis.set_major_formatter(formatt)
    plt.xticks(rotation = '90')
    plt.show()
    

def spearman(data, labels, cm="seismic", fons=20, figs=(20, 17), tfs=25, ti="Spearman"):
    """
    Calculates Pearson matrix and plots it.

    :type data: list[float]
    :param data: Data to correlate.

    :type labels: list[str]
    :param labels: Labels of the data.

    :type cm: str
    :param cm: Color map as for matplolib. The default is "seismic".

    :type fons: int
    :param fons: Plot font size. The default is 20.

    :type figs: tuple
    :param figs: Plot size. The default is (20,17).

    :type tfs: int
    :param tfs: Title font size. The default is 25.

    :type ti: str
    :param ti: Plot title/name. The default is "spearman".
    """
    data = list(np.transpose(data))
    labels = list(labels)
    cm = str(cm)
    fonsize = float(fons)
    figsize = list(figs)
    titlefs = float(tfs)
    title = str(ti)
    n = len(data)
    
    spear = [] # spearman
    cordsi = [] # coordinates, same order as labels
    cordsj = [] # coordinates, same order as labels
    for i in range(n):
        for j in range(n):# compare with all sets
            x = data[i]
            y = data[j]
            spear.append(stats.spearmanr(x,y)[0])
            
            cordsi.append(int(i))
            cordsj.append(int(j))
    
    merged_spear = np.c_[ spear, cordsi, cordsj ] # merge to sort together    
    merged_spear = sorted(merged_spear,key=lambda l:l[0], reverse=True) 
    
    for i in range(n): # delete the first n (the obvious 1s)
        merged_spear = np.delete(merged_spear, 0, 0)
    
    for i in range(int((n*n-n)/2)): # delete the repeated half
        merged_spear = np.delete(merged_spear, i, 0)
    
    spear = np.reshape(spear,(n,n)) #[rho spearman, p-value]

    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])
    plt.rc('font', size=fonsize)
    fig = plt.figure(tight_layout=True,figsize=figsize)
    
    y = [i+0.5 for i  in range(n)]
    ticks = mpl.ticker.FixedLocator(y)
    formatt = mpl.ticker.FixedFormatter(labels)
    
    ax = fig.add_subplot(gs[0, 0])
    pcm = ax.pcolormesh(spear, cmap=cm, vmin=-1, vmax=1)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(title, fontsize=titlefs)
    ax.xaxis.set_major_locator(ticks)
    ax.yaxis.set_major_locator(ticks)
    ax.xaxis.set_major_formatter(formatt)
    ax.yaxis.set_major_formatter(formatt)
    plt.xticks(rotation = '90')
    plt.show()


def grau(data, labels, cm="seismic", fons=20, figs=(25, 15),
             tfs=25, ti="Grau (Beta)", marker="s", marks=100):
    """
    Performs Grau correlation matrix and plots it.

    :type data: list[float]
    :param data: Data to be correlated.

    :type labels: list[str]
    :param labels: Labels of the data to be ploted.

    :type cm: str
    :param cm: Color map for the plot, from matplotlib. The default is "seismic".

    :type fons: int, optional
    :param fons: Plot font size. The default is 20.

    :type figs: tuple
    :param figs: Figure size. The default is (25,15).

    :type tfs: int
    :param tfs: Plot title font size. The default is 25.

    :type ti: str
    :param ti: Plot title. The default is "Grau (Beta)".

    :type marker: str
    :param marker: Plot marker type (scatter). The default is "s".

    :type marks: int
    :param marks: DESCRIPTION. The default is 100.
    """
    data = list(np.transpose(data))
    labels = list(labels)
    cm = str(cm)
    fonsize = float(fons) # plot font size
    figsize = list(figs) # plot size
    titlefs = float(tfs) # title font size
    title = str(ti) # plot name
    marker = str(marker) # market style
    markersize = float(marks) # marker size
    n = len(data)
    
    graus = evalgrau(data) # grau correlation (3d R2)
    g1 = [graus[i][0] for i in range(len(graus))] # first dimeniosn values
    g2 = [graus[i][1] for i in range(len(graus))] # second dimension values
    g3 = [graus[i][2] for i in range(len(graus))] # third dimension values
    mse = [graus[i][3] for i in range(len(graus))] # mse's list
    g2_shift = list(g2) # copy v2 to displace the d2 values for plotting
    t_c = [] # list of ticks per subplot
    xtick_labels = [] # list for labels
    c = 1 # number of different # in combs[i][0] (first column), starts with 1
    
    for i in range(len(graus) - 1): # for all combinatios
        if graus[i][0] != graus[i + 1][0]: # if it is different
            c += 1 # then it is a new one
    
    for i in range(c): # for all the different first positions
        temp = 0 # temporal variable to count ticks
        for j in range(len(graus) - 1): # check all the combinations
            if graus[j][0] == i: # if it is the one we are looking for
                if  graus[j][1] != graus[j + 1][1]: # if it changes, then new tick
                    temp +=1 # add
        if temp == 0: # if it doesnt count, is because there is only 1
            t_c.append(1) # so append 1
        elif temp == 1: # if only 1 is counted
            t_c.append(2) # then add a 2
        else: # otherwise, 
            t_c.append(temp) # when it is done, append to the list    
    
    for i in range(len(t_c)): # for all the different #
        for j in range(t_c[i] + 1): # for the number of different 2d
            xtick_labels.append(j + i) # append the value +1
    xtick_labels.append('') # append '' for the last label to be blank
    
    for i in range(len(g1)): # to shift the position for plotting
        for j in range(1,len(t_c)): # for all the ticks in x axis
            if g1[i] >= j: # if bigger than 0 (first)
                g2_shift[i] += t_c[j-1] # shift (add) all the previous
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1]) # (rows, columns)
    plt.rc('font', size=fonsize)
    fig = plt.figure(tight_layout=True,figsize=figsize)
    
    y = [i+0.5 for i  in range(n)]
    
    ax = fig.add_subplot(gs[0, 0])
    cm = plt.cm.get_cmap(cm)
    ax.set_title(title, fontsize=titlefs)
    sc = ax.scatter(g2_shift,g3,alpha=1,edgecolors='none',
                    c=mse,cmap=cm,s=markersize,marker=marker)
    plt.colorbar(sc)
    y_ticks = [i for i in range(int(min(g3)),int(max(g3)) + 1)]
    x_ticks = [i for i in range(int(max(g2_shift)) + 2)]
    ytick_labels = []
    for i in range(len(y_ticks)):
        ytick_labels.append(labels[y_ticks[i]])
        
    for i in range(len(xtick_labels)-1):
        xtick_labels[i] = labels[xtick_labels[i]]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlim(0, max(x_ticks))
    ax.get_xticklabels()[0].set_fontweight('bold')
    plt.xticks(rotation = '90')
    ax.set_yticks(y_ticks)
    ax.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
    ax.set_yticklabels(ytick_labels, fontsize=20)
    ax.grid(linestyle='--')
    
    temp = 0
    for i in range(len(t_c)):
        temp += t_c[i] + 1
        ax.get_xgridlines()[temp].set_linestyle('-')
        ax.get_xgridlines()[temp].set_color('black')
        ax.get_xgridlines()[temp].set_linewidth(1)
        ax.get_xticklabels()[temp].set_fontweight('bold')
    
    plt.show()

def moveavg(data, move):
    """
    Calculate the moving average of a single or multiple vectors.

    :type move: int
    :param move: Average range to each side (total average = move + 1).

    :type data: list[float]
    :param data: Data to calculate the moving averge. Single or multiple vectors.

    :returns: Smoothed vector(s).
    :rtype: list[float]
    """
    move = int(move)
    b_data = copy.deepcopy(data)
    
    dims = len(np.array(b_data).shape)

    avg = [] # for smoothed data
    
    if dims >= 2:   
        data_len = len(b_data[0])
    
        for j in range(len(b_data)): # each measured point
            temp = []
            for i in range(move, data_len - move):
                temp.append(np.mean(b_data[j][i - move: i + move + 1]))        
            for i in range(move):
                temp.append(0)
                temp.insert(0, 0)  
            avg.append(temp)  
    else:
        data_len = len(b_data)      
        
        for i in range(move, data_len - move):
            avg.append(np.mean(b_data[i - move: i + move + 1]))      
        for i in range(move):
            avg.append(0)
            avg.insert(0, 0)  
                
    return avg
