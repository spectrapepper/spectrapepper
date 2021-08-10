"""
This main module contains all the functions available in spectrapepper. Please
use the search function to look up specific functionalities with keywords.
"""

import math
import copy
import random
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import interpolate
from scipy.stats import stats
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
from scipy.interpolate import splev, splrep
from scipy.sparse.linalg import spsolve
import itertools
from itertools import combinations
import statistics as sta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import linecache
import os.path
import os


def load_mapp1():
    """
    Load sample specrtal data, axis included in the first line.

    :returns: Sample spectral data.
    :rtype: list[float]
    """
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'mapping1.txt')
    data = load(my_file)

    return data


def load_mapp2():
    """
    Load sample specrtal data, axis included in the first line.

    :returns: Sample spectral data.
    :rtype: list[float]
    """
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'mapping2.txt')
    data = load(my_file)

    return data


def load_spectras():
    """
    Load sample specrtal data, axis included in the first line.
    
    :returns: Sample spectral data.
    :rtype: list[float]
    """
    # path = os.getcwd()
    # parent = os.path.abspath(os.path.join(path, os.pardir))
    # data = load(parent+'\spectrapepper\datasets\spectras.txt')
    
    # module_path = os.path.dirname(__file__)
    # data = load(module_path+'\\datasets\\spectras.txt')
    
    # path = str(PACKAGEDIR)+'spectrapepper\datasets\spectras.txt'
    # print(path)
    # text = load(path)
    # print(text)
    
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'spectras.txt')
    data = load(my_file)
    
    return data


def load_targets():
    """
    Load sample targets data for the spectras.
    
    :returns: Sample targets.
    :rtype: list[float]
    """
    # path = os.getcwd()
    # parent = os.path.abspath(os.path.join(path, os.pardir))
    # data = load(parent+'\spectrapepper\datasets\targets.txt')    
    
    # module_path = os.path.dirname(__file__)
    # data = load(module_path+'\\datasets\\targets.txt')
    
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'targets.txt')
    data = load(my_file)
    
    return data


def load_params(transpose=False):
    """
    Load sample parameters data for the spectras.
    
    :returns: Sample parameters.
    :rtype: list[float]
    """
    # path = os.getcwd()
    # parent = os.path.abspath(os.path.join(path, os.pardir))
    # data = load(parent+'\spectrapepper\datasets\params.txt')   
    
    # module_path = os.path.dirname(__file__)
    # data = load(module_path+'\\datasets\\params.txt')
    
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'params.txt')
    data = load(my_file)
    
    if transpose:
        data = np.transpose(data)
    
    return data


def load(file, fromline=0, transpose=False, dtype=float):
    """
    Load data from a standard text file obtained from LabSpec and other
    spectroscopy instruments. Normally, when single measurement these come in 
    columns with the first one being the x-axis. When it is a mapping, the
    first row is the x-axis and the following are the measurements.

    :type file: str
    :param file: Url of data file. Must not have headers and separated by 'spaces' (LabSpec).
    
    :type fromline: int
    :param fromline: Line of file from which to start loading data. The default is 0.
    
    :type transpose: boolean
    :param transpose: If True transposes the data. Default is False.
    
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
            s_row = np.array(s_row, dtype=dtype)
            new_data.append(s_row)
        i += 1
    raw_data.close()

    if transpose:
        new_data = np.transpose(new_data)

    return new_data


def loadline(file, line=0, dtype='float', split=False):
    """
    Random access to file. Loads a specific line in a file. Useful when
    managing large data files in processes where time is important. It can
    load numbers as floats.

    :type file: str
    :param file: Url od the data file

    :type line: int
    :param line: Line number. Counts from 0.

    :type dtype: str
    :param dtype: Type of data. If its numeric then 'float', if text then 'string'.
        Default is 'float'.

    :type split: boolean
    :param split: True to make a list of strings when 'tp' is 'string', 
        separated by space. The default is False.

    :returns: Array with the desired line.
    :rtype: list[float]
    """
    line = int(line) + 1
    file = str(file)
    info = linecache.getline(file, line)
    
    if dtype == 'float':
        info = info.replace("NaN", "-1")
        info = info.replace("nan", "-1")
        info = info.replace("--", "-1")
        info = str.split(info)
        info = np.array(info, dtype=float)

    if dtype == 'string':
        if split:
                info = str.split(info)
        info = np.array(info, dtype=str)

    return info


def test_loads():
    """
    Only for tests purposes. Ingroed in documentation.
    
    :returns: A True is test is succesful.
    :rtype: bool
    """
    default = False
    
    location = os.path.dirname(os.path.realpath(__file__))
    
    my_file = os.path.join(location, 'datasets', 'spectras.txt')
    data = loadline(my_file,2)
    r = round(sum(data), 0)
    print(r)
    if r == 1461:
        default = True
    
    my_file = os.path.join(location, 'datasets', 'headers.txt')
    data = loadline(my_file, 1, dtype='string', split=True)
    r = data[2]
    if r != 'second':
        default = False
    
    return default


def lowpass(data, cutoff=0.25, fs=30, order=2, nyq=0.75):
    """
    Butter low pass filter for a single or spectra or a list of them.
        
    :type data: list[float]
    :param data: List of vectors in line format (each line is a vector).
    
    :type cutoff: float
    :param cutoff: Desired cutoff frequency of the filter. The default is 0.25.
    
    :type fs: int
    :param fs: Sample rate in Hz. The default is 30.
    
    :type order: int
    :param order: Sin wave can be approx represented as quadratic. The default is 2.
    
    :type nyq: float
    :param nyq: Nyquist frequency, 0.75*fs is a good value to start. The default is 0.75*30.

    :returns: Filtered data
    :rtype: list[float]
    """
    y = copy.deepcopy(data)  # so it does not change the input list
    normal_cutoff = cutoff / (nyq * fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(np.array(y).shape) > 1:
        for i in range(len(y)):
            y[i] = filtfilt(b, a, y[i])
    else:
        y = filtfilt(b, a, y)
    return y


def normtomax(data, zeromin=False):
    """
    Normalizes spectras to the maximum value of each, in other words, the
    maximum value of each spectras is set to 1.

    :type data: list[float]
    :param data: Single or multiple vectors to normalize.
        
    :type zeromin: boolean
    :param zeromin: If `True`, the minimum value is traslated to 0. Default
        values is `False`
    
    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(data)  # so it does not chamge the input list
    dims = len(np.array(y).shape)  # detect dimensions
    if dims > 1:
        for i in range(len(y)):           
            if zeromin:
                min_data = min(y[i])
                y[i] = y[i] - min_data
            max_data = max(y[i])
            y[i] = y[i]/max_data
    else:
        if zeromin:
            min_data = min(y)
            y = y -  min_data
        max_data = max(y)
        y = y/max_data
    return y


def normtovalue(data, val):
    """
    Normalizes the spectras to a set value, in other words, the defined value
    will be reescaled to 1 in all the spectras.

    :type data: list[float]
    :param data: Single or multiple vectors to normalize.

    :type val: float
    :param val: Value to normalize to.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(data)  # so it does not change the input list
    y = np.array(y)/val
    y = list(y)
    return y


def alsbaseline(data, lam=100, p=0.001, niter=10):
    """
    Calculation of the baseline using Asymmetric Least Squares Smoothing. This
    script only makes the calculation but it does not remove it. Original idea of
    this algorithm by P. Eilers and H. Boelens (2005):

    :type data: list[float]
    :param data: Spectra to calculate the baseline from.

    :type lam: int
    :param lam: Lambda, smoothness. The default is 100.

    :type p: float
    :param p: Asymmetry. The default is 0.001.

    :type niter: int
    :param niter: Niter. The default is 10.

    :returns: Returns the calculated baseline.
    :rtype: list[float]
    """
    data = copy.deepcopy(data)
    dims = len(np.array(data).shape)  # detect dimensions    
    
    if dims > 1:
        l = len(data[0])
        d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(l, l - 2))
        w = np.ones(l)
        for i in range(len(data)):
            for _ in range(niter):
                W = sparse.spdiags(w, 0, l, l)
                Z = W + lam * d.dot(d.transpose())       
                z = spsolve(Z, w * data[i])
                w = p * (data[i] > z) + (1 - p) * (data[i] < z)
            data[i] = data[i] - z  
            
    else:
        l = len(data)
        d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(l, l - 2))
        w = np.ones(l)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, l, l)
            Z = W + lam * d.dot(d.transpose())
            z = spsolve(Z, w * data)
            w = p * (data > z) + (1 - p) * (data < z)
        data = data - z
    return data


def bspbaseline(data, x_axis, points, avg=5, remove=True, plot=False):
    """
    Calcuates the baseline using b-spline.

    :type data: list[float]
    :param data: Single or several spectras to remove the baseline from.

    :type x_axis: list[float]
    :param x_axis: x axis of the data, to interpolate the baseline function.

    :type points: list[int]
    :param points: axis values of points to calculate the bspine.

    :type avg: int
    :param avg: points to each side to make average.

    :type remove: True
    :param remove: if True, calculates and returns (data - baseline).

    :type plot: bool
    :param plot: if True, calculates and returns (data - baseline).    

    :returns: The baseline.
    :rtype: list[float]
    """
    data = copy.deepcopy(data)
    x_axis = list(x_axis)
    x = list(points)
    avg = int(avg)
    pos = cortopos(x, x_axis)
    dims = len(np.array(data).shape)  # detect dimensions

    if dims >= 2:
        baseline = []
        
        for j in range(len(data)):
            y = []  # y values for the selected x
            for i in range(len(pos)):
                temp = np.mean(data[j][pos[i] - avg: pos[i] + avg + 1])
                y.append(temp)
        
            spl = splrep(x, y)
                
            if remove:
                baseline.append(data[j] - splev(x_axis, spl))
            else:
                baseline.append(splev(x_axis, spl))

    else:
        y = []  # y values for the selected x
        for i in range(len(pos)):
            temp = np.mean(data[pos[i] - avg: pos[i] + avg + 1])
            y.append(temp)
    
        spl = splrep(x, y)
        baseline = splev(x_axis, spl)
        
        if plot:        
            plt.plot(x_axis, data)
            plt.plot(x_axis, baseline)
            plt.plot(x, y, 'o', color='red')
            plt.show()
    
        if remove:
            baseline = data - baseline

    return baseline


def polybaseline(data, x_axis, points, deg=2, avg=5, remove=True, plot=False):
    """
    Calcuates the baseline using polynomial fit.

    :type data: list[float]
    :param data: Single or several spectras to remove the baseline from.

    :type x_axis: list[float]
    :param x_axis: x axis of the data, to interpolate the baseline function.

    :type points: list[int]
    :param points: positions in axis of points to calculate baseline.

    :type deg: int
    :param deg: Polynomial degree of the fit.

    :type avg: int
    :param avg: points to each side to make average.

    :type remove: True
    :param remove: if True, calculates and returns (data - baseline).

    :type plot: bool
    :param plot: if True, calculates and returns (data - baseline).    

    :returns: The baseline.
    :rtype: list[float]
    """
    data = copy.deepcopy(data)  
    x_axis = list(x_axis)
    x = list(points)
    avg = int(avg)
    pos = cortopos(x, x_axis)
    dims = len(np.array(data).shape)  # detect dimensions
    
    if dims > 1:
        baseline = []
        for j in range(len(data)):
            y = []  # y values for the selected x
            temp = []
            for i in range(len(pos)):
                temp = np.mean(data[j][pos[i] - avg: pos[i] + avg + 1])
                y.append(temp)
                        
            z = np.polyfit(x, y, deg)  # polinomial fit
            f = np.poly1d(z)  # 1d polinomial
            temp = f(x_axis)  # y values
            if plot and j == 0:        
                plt.plot(x_axis, data[j])
                plt.plot(x_axis, temp)
                plt.plot(x, y, 'o', color='red')
                plt.show()
        
            if remove:
                temp = data[j] - temp
            baseline.append(temp)
    
    else:
        y = []  # y values for the selected x
        for i in range(len(pos)):
            temp = np.mean(data[pos[i] - avg: pos[i] + avg + 1])
            y.append(temp)
    
        z = np.polyfit(x, y, deg)  # polinomial fit
        f = np.poly1d(z)  # 1d polinomial
        baseline = f(x_axis)  # y values
    
        if plot:        
            plt.plot(x_axis, data)
            plt.plot(x_axis, baseline)
            plt.plot(x, y, 'o', color='red')
            plt.show()
    
        if remove:
            baseline = data - baseline

    return baseline


def lorentzfit(y, x, wid=4, it=100, plot=False):
    """
    Fit a Lorentz distributed curve for a single spectra.

    :type y: list[float]
    :param y: single spectra.

    :type x: list[float]
    :param x: x-axis.

    :type wid: float, optional
    :param wid: fitted curve width. The default is 4.

    :type it: int, optional
    :param it: number of iterations. The default is 100.

    :type plot: boolean
    :param plot: fF 'True' then plots the data and the fit.

    :returns: Lorentz fit.
    :rtype: list[float]
    """
    cen = 0  # position (wavelength, x value)
    peak_i_pos = 0  # position (axis position, i value)
    amp = max(y)  # maximum value of function, peak value, amplitud of the fit
    #  fit = []  # fit function
    err = []  # error log to choose the optimum
    for i in range(len(y)):  # for all the data
        if amp == y[i]:  # if it is the maximum
            cen = x[i]  # save the axis value x
            peak_i_pos = int(i)  # also save the position i
            break
    for i in range(it):  # search "it" iterations
        temp = 0  # reset temporal error
        fit = []  # reset lorentz fit array
        p = cen - 0.5 + (0.01 * i)  # pos of the peak respect to the reference measured peak
        for j in range(len(x)):  # for all the points
            fit.append(amp * wid ** 2 / ((x[j] - p) ** 2 + wid ** 2))
        for j in range(peak_i_pos - 25, peak_i_pos + 25):  # error between -25 and +25 of the peak position
            temp = temp + (fit[j] - y[j]) ** 2  # total error
        err.append(temp)  # log error
    fit = []  # reset array
    for i in range(len(err)):  # look for the minimum error
        if min(err) == err[i]:  # if it is the minimum error
            p = cen - 0.5 + (0.01 * i)  # then calculate the fit again
            for j in range(len(y)):
                fit.append(amp * wid ** 2 / ((x[j] - p) ** 2 + wid ** 2))
            break

    if plot:
        plt.plot(x, y, label='Data')
        plt.plot(x, fit, label='Fit')
        plt.legend(loc=0)
        plt.show()

    return fit


def gaussfit(y, x, sigma=4.4, it=100, plot=False):
    """
    Fit a Gaussian distributed curve for a single spectra. Good guidelines
    for similar structures can be found at http://emilygraceripka.com/blog/16.

    :type y: list[float]
    :param x: single spectra.

    :type x: list[float]
    :param x: x-axis.

    :type sigma: float
    :param sigma: sigma parameter of distribution. The default is 4.4.

    :type it: int
    :param it: number of iterations. The default is 100.

    :type plot: boolean
    :param plot: fF 'True' then plots the data and the fit.

    :returns: Gaussian fit.
    :rtype: list[float]
    """
    # x = list(ax)  # change the variable so it doesnt change the original
    # y = list(data_x)  # change the variable so it doesnt change the original
    cen = 0  # position (wavelength, x value)
    peak_i_pos = 0  # position (axis position, i value)
    amp = max(y)  # maximum value of function, peak value, amplitude of the fit
    # fit = []  # fit function
    err = []  # error log to choose the optimum
    for i in range(len(y)):  # for all the data
        if amp == y[i]:  # if it is the maximum
            cen = x[i]  # save the axis value x
            peak_i_pos = int(i)  # also save the position i
            break
    for i in range(it):  # search "it" iterations
        temp = 0  # reset temporal error
        fit = []  # reset lorentz fit array
        p = cen - 0.5 + (0.01 * i)  # pos of the peak respect to the reference measured peak
        for j in range(len(x)):  # for all the points
            fit.append((11 * amp) * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp(-0.5 * (((x[j] - p) / sigma) ** 2))))
        for j in range(peak_i_pos - 25, peak_i_pos + 25):  # error between -25 and +25 of the peak position
            temp = temp + (fit[j] - y[j]) ** 2  # total error
        err.append(temp)  # log error
    fit = []  # reset array
    for i in range(len(err)):  # look for the minimum error
        if min(err) == err[i]:  # if it is the minimum error
            p = cen - 0.5 + (0.01 * i)  # then calculate the fit again
            for j in range(len(y)):
                fit.append(
                    (11 * amp) * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp(-0.5 * (((x[j] - p) / sigma) ** 2))))
            break

    if plot:
        plt.plot(x, y, label='Data')
        plt.plot(x, fit, label='Fit')
        plt.legend(loc=0)
        plt.show()

    return fit


def cortopos(vals, axis):
    """
    To translate values to a position in an axis, basically searching for
    the position of a value. Normally they don't fit perfectly, so it is useful
    to use this tool that approximates to the closest.

    :type vals: list[float]
    :param vals: List of values to be searched and translated.

    :type axis: list[float]
    :param axis: Axis.

    :returns: Position in the axis of the values in vals
    :rtype: list[int]
    """
    if len(np.array(vals).shape) > 1:
        pos = [[0 for _ in range(len(vals[0]))] for _ in range(len(vals))]  # i position of area limits
        for i in range(len(vals)):  # this loop takes the approx. x and takes its position
            for j in range(len(vals[0])):
                dif_temp = 999  # safe initial difference
                temp_pos = 0  # temporal best position
                for k in range(len(axis)):  # search in x_axis
                    if abs(vals[i][j] - axis[k]) < dif_temp:  # compare if better
                        temp_pos = k  # save best value
                        dif_temp = abs(vals[i][j] - axis[k])  # calculate new diff
                vals[i][j] = axis[temp_pos]  # save real value in axis
                pos[i][j] = temp_pos  # save the position
    else:
        pos = []  # i position of area limits
        for i in range(len(vals)):  # this loop takes the approx. x and takes its position
            dif_temp = 999  # safe initial difference
            temp_pos = 0  # temporal best position
            for k in range(len(axis)):
                if abs(vals[i] - axis[k]) < dif_temp:
                    temp_pos = k
                    dif_temp = abs(vals[i] - axis[k])
            vals[i] = axis[temp_pos]  # save real value in axis
            pos.append(temp_pos)  # save the position
    return pos


def areacalculator(data, limits, norm=False):
    """
    Area calculator using the data (x_data) and the limits in position, not
    values.

    :type data: list[float]
    :param data: Data to calculate area from

    :type limits: list[int]
    :param limits: Limits that define the areas to be calculated. Axis position.
    
    :type norm: bool
    :param norm: If True, normalized the area to the sum under all the curve.

    :returns: A list of areas according to the requested limits.
    :rtype: list[float]
    """
    dims = len(np.array(data).shape)

    if dims >= 2:
        areas = [[0 for _ in range(len(limits))] for _ in range(len(data))]  # final values of areas
        for i in range(len(data)):  # calculate the areas for all the points
            for j in range(len(limits)):  # for all the areas
                areas[i][j] = np.sum(data[i][limits[j][0]:limits[j][1]])  # calculate the sum
                if norm:
                    areas[i][j] = areas[i][j] / np.sum(data[i])
    else:
        areas = [0 for _ in range(len(limits))]  # final values of areas
        for j in range(len(limits)):  # for all the areas
            areas[j] = np.sum(data[limits[j][0]:limits[j][1]])  # calculate the sum
            if norm:
                areas[j] = areas[j] / np.sum(data)
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

    iters = []  # final matrix
    temp = []  # temp matrix
    stuff = []  # matrix of ALL combinations

    for i in range(n):  # create possibilities vector (0 and 1)
        stuff.append(1)
        stuff.append(0)

    for subset in itertools.combinations(stuff, n):
        temp.append(subset)  # get ALL combinations possible from "stuff"

    for i in range(len(temp)):  # for all the possibilities...
        e = 0
        for j in range(len(iters)):  # check if it already exists
            if temp[i] == iters[j]:  # if it matches, add 1
                e += 1
            else:
                e += 0  # if not, add 0
        # e == 0 and sum(temp[i]) >= s_min and sum(temp[i]) <= s_max
        if e == 0 and s_min <= sum(temp[i]) <= s_max:
            iters.append(temp[i])  # if new and fall into the criteria, then add

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
    if dims > 1:
        for i in range(len(y)):
            s = sum(y[i])
            for j in range(len(y[i])):
                y[i][j] = y[i][j] / s
    else:
        y = list(data)
        s = sum(y)
        for i in range(len(y)):
            y[i] = y[i] / s
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
    if dims > 1:
        maximum = -99999999  # safe start
        for i in range(len(y)):
            for j in range(len(y[0])):
                if y[i][j] > maximum:
                    maximum = y[i][j]
        y = normtovalue(y, maximum)
    else:  # if s single vector, then is the same as nortomax (loca)
        y = normtomax(y)
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
    temp_y = list(data)  # data, no axis
    master_x = list(x_axis)  # axis

    # NEW AXIS
    new_step = 1
    new_start = -1
    new_end = 99999999
    for i in range(len(master_x)):
        if min(master_x[i]) > new_start:
            new_start = math.ceil(min(master_x[i]))
        if max(master_x[i]) < new_end:
            new_end = math.floor(max(master_x[i]))
    x_new = np.arange(new_start, new_end + new_step, new_step)

    # MASTER DATA
    master_y = []
    for i in range(len(temp_y)):
        for j in range(len(temp_y[i])):
            this = interpolate.interp1d(master_x[i], temp_y[i][j])
            master_y.append(this)

    # INTERPOLATIONS
    for i in range(len(master_y)):
        master_y[i] = master_y[i](x_new)

    return master_y, x_new


def evalgrau(data):
    """
    This function evaluates the MSE in 3 dimensions (x,y,z) for a set of
    data vectors.

    :type data: list[float]
    :param data: A list of lists of variables to compare.

    :returns: A list with each combination and the R2 score obtained.
    :rtype: list[float]
    """
    data = list(data)
    tup = [i for i in range(len(data))]  # number list of data type sets
    combs = tuple(combinations(tup, 3))  # make all combinations
    n_o_p = len(data[0])

    r2 = []  # R2 list
    for i in range(len(combs)):  # for all the combinations
        xs = data[combs[i][0] - 1]  # temporal x axis
        ys = data[combs[i][1] - 1]  # temporal y axis
        zs = data[combs[i][2] - 1]  # temporal z axis
        a = []  # fit function parameter
        b = []  # fit function parameter
        for j in range(n_o_p):  # for all the data points
            a.append([xs[j], ys[j], 1])  # A value
            b.append(zs[j])  # b value
        b = np.matrix(b).T  # transpose matrix form
        a = np.matrix(a)  # matrix form
        fit = (a.T * a).I * a.T * b  # evaluate fir
        rss = 0  # residual sum of squares
        tss = 0  # total sum of squares

        for j in range(n_o_p):  # calculate mse for all the points
            rss += (zs[j] - (xs[j] * fit[0] + ys[j] * fit[1] + fit[2])) ** 2  # residual sum
            tss += (zs[j] - np.mean(zs)) ** 2  # total error
        r2.append(round(float(1 - rss / tss), 2))
    merged = np.c_[combs, r2]

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
    g_count = [0 for _ in range(int(max(all_targets) + 1))]  # list to count points per group
    g_scores = [0 for _ in range(int(max(all_targets) + 1))]  # list to store the scores
    for i in range(len(g_scores)):  # for all the groups
        for j in range(len(predicted_targets)):  # for all the points
            if used_targets[j] == i:
                g_count[i] += 1
                if predicted_targets[j] == used_targets[j]:
                    g_scores[i] += 1
    for i in range(len(g_scores)):
        g_scores[i] = round(g_scores[i] / g_count[i], 2)
    return g_scores


def cmscore(x_points, y_points, target):
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

    :returns: Score by comparing CM distances. Prediction using CM distances. X-axis coords of ths CMs.
        Y-axis coords of the Cms.
    :rtype: list[float, list[int],list[float],list[float]]
    """
    x_p = list(x_points)
    y_p = list(y_points)
    tar = list(target)
    g_n = int(max(target) + 1)

    a = [0 for _ in range(g_n)]  # avg D1
    b = [0 for _ in range(g_n)]  # avg D2
    c = [0 for _ in range(g_n)]  # N for each group
    d = []  # distances
    p = []  # predictions

    for i in range(len(tar)):
        for j in range(g_n):
            if tar[i] == j:
                a[j] += x_p[i]
                b[j] += y_p[i]
                c[j] += 1

    for i in range(g_n):
        a[i] = a[i] / c[i]
        b[i] = b[i] / c[i]

    correct = 0
    for i in range(len(tar)):
        temp1 = -1
        temp2 = 1000
        temp3 = []

        for j in range(g_n):
            temp3.append(((x_p[i] - a[j]) ** 2
                          + (y_p[i] - b[j]) ** 2) ** 0.5)

            if temp3[j] < temp2:
                temp2 = temp3[j]
                temp1 = j

        p.append(temp1)
        d.append(temp3)

        if tar[i] == temp1:
            correct += 1

    score = round(correct / len(tar), 2)

    return score, p, a, b


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

    :returns: Score by comparing MD distances. Prediction using MD distances. X-axis coords of ths CMs.
        Y-axis coords of the Cms.
    :rtype: list[float, list[int],list[float],list[float]]
    """
    x_p = list(x_p)
    y_p = list(y_p)
    g_n = max(tar) + 1
    tar = list(tar)

    a = [0 for _ in range(g_n)]
    b = [0 for _ in range(g_n)]
    d = []  # distances
    p = []  # predictions

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
            temp3.append(((x_p[i] - a[j]) ** 2
                          + (y_p[i] - b[j]) ** 2) ** 0.5)

            if temp3[j] < temp2:
                temp2 = temp3[j]
                temp1 = j

        p.append(temp1)
        d.append(temp3)

        if tar[i] == temp1:
            correct += 1

    score = round(correct / len(tar), 2)

    return score, p, a, b


def normtopeak(data, x_axis, peak, shift=10):
    """
    Normalizes the spectras to a particular peak.

    :type data: list[float]
    :param data: Data to be normalized.

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
    dims = len(np.array(y).shape)
    
    x_axis = list(x_axis)
    # peak = [int(peak)]
    shift = int(shift)
    pos = cortopos([int(peak)], x_axis)
    
    if dims > 1:
        for j in range(len(y)):
            section = y[j][pos[0] - shift:pos[0] + shift]
            highest = peakfinder(section, look=int(shift / 2))
        
            c = 0
            for i in range(len(highest)):
                if highest[i] == 1:
                    c = i
                    break
        
            local_max = y[j][pos[0] - shift + c]
            y[j] = y[j] / local_max
    else:
        section = y[pos[0] - shift:pos[0] + shift]
        highest = peakfinder(section, look=int(shift / 2))
    
        c = 0
        for i in range(len(highest)):
            if highest[i] == 1:
                c = i
                break
    
        local_max = y[pos[0] - shift + c]
        y = y / local_max
    return y


def peakfinder(data, look=10):
    """
    Find the location of the peaks in a single vector.

    :type data: list[float]
    :param data: Data to find a peak in.

    :type look: int
    :param look: Amount of position to each side to decide if it is a local maximum. The default is 10.

    :returns: A vector of same length with 1 or 0 if it finds or not a peak in that position
    :rtype: list[int]
    """
    y = copy.deepcopy(data)
    is_max = [0 for _ in data]
    is_min = [0 for _ in data]
    for i in range(look, len(y) - look):  # start at "look" to avoid o.o.r
        lower = 0  # negative if lower, positive if higher
        higher = 0
        for j in range(look):
            if y[i] <= y[i - look + j] and y[i] <= y[i + j]:  # search all range lower
                lower += 1  # +1 if lower
            elif (y[i] >= y[i - look + j] and
                  y[i] >= y[i + j]):  # search all range higher
                higher += 1  # +1 if higher
        if higher == look:  # if all higher then its local max
            is_max[i] = 1
            is_min[i] = 0
        elif lower == look:  # if all lower then its local min
            is_max[i] = 0
            is_min[i] = 1
        else:
            is_max[i] = 0
            is_min[i] = 0
    return is_max


def confusionmatrix(tt, tp, gn=['', '', ''], plot=False, title='', 
                    cmm='Blues', fontsize=20, ylabel='True', xlabel='Prediction'):
    """
    Calculates and/or plots the confusion matrix for machine learning algorithm results.

    :type tt: list[float]
    :param tt: Real targets.

    :type tp: list[float]
    :param tp: Predicted targets.

    :type plot: boolean
    :param plot: If true, plots the matrix. The default is False

    :type gn: list[str]
    :param gn: Names, or lables , of the classification groups. Default is empty.

    :type title: str
    :param title: Name of the matrix. Default is empty.

    :type cmm: str
    :param cmm: Nam eof the colormap (matplotlib) to use for the plot. Default is Blue.

    :type fontsize: int
    :param fontsize: Font size for the labels. The default is 20.

    :type ylabel: str
    :param ylabel: Label for y axis. Default is `True`.
    
    :type xlabel: str
    :param xlabel: Label for x axis. Default is `Prediction`.
    
    :returns: The confusion matrix
    :rtype: list[float]
    """
    tt = list(tt)
    tp = list(tp)
    plot = bool(plot)
    group_names = list(gn)
    gn = len(group_names)
    title = str(title)
    cmm = str(cmm)
    fontsize = int(fontsize)

    m = [[0 for _ in range(gn)] for _ in range(gn)]
    p = [0 for _ in range(gn)]

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
        fig = plt.figure(tight_layout=True, figsize=(6, 7.5))
        plt.rc('font', size=fontsize)
        ax = fig.add_subplot()
        ax.imshow(m, cmap=cmm)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(group_names)))
        ax.set_yticks(np.arange(len(group_names)))
        ax.set_xticklabels(group_names)
        ax.set_yticklabels(group_names)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(gn):
            for j in range(gn):
                ax.text(j, i, round(m[i][j], 2), ha='center', va='center', color='black')
        
        plt.show()
        
    return m


def avg(data):
    """
    Calculates the average vector from a list of vectors.

    :type data: list[float]
    :param data: List of vectors.

    :returns: The average of the vectors in the list.
    :rtype: list[float]
    """
    data = list(data)
    avg_data = np.array([0 for _ in range(len(data[0]))])
    for i in data:
        avg_data = avg_data + i
    avg_data = avg_data / len(data)
    return avg_data


def sdev(data):
    """
    Calculates the standard deviation for each bin from a list of vectors.

    :type data: list[float]
    :param data: List of vectors.

    :returns: Standard deviation curve
    :rtype: list[float]
    """
    curve_std = []  # stdev for each step
    data = list(data)
    for i in range(len(data[0])):
        temp = []
        for j in range(len(data)):
            temp.append(data[j][i])
        curve_std.append(sta.stdev(temp))  # stdev for each step
    curve_std = np.array(curve_std)
    return curve_std


def median(data):
    """
    Calculates the median vector of a list of vectors.
    
    :type data: list[float]
    :param data: List of vectors.

    :returns: median curve
    :rtype: list[float]    
    """
    median = []
    length = len(data[0])
    meas = len(data)
    
    for j in range(length):
        temp = []
        for i in range(meas):
            temp.append(data[i][j])
        median.append(np.median(temp))    
    return median


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
    amp = max(y) / 2
    wid = float(wid)

    def objective(x):
        fit = []
        error = 0
        for i in range(len(ax)):  # for all the points
            fit.append(x[0] * x[1] ** 2 / ((ax[i] - ax[pos]) ** 2 + x[1] ** 2))
        for j in range(pos - look, pos + look):  # error between +-look of the peak position
            error += (fit[j] - y[j]) ** 2  # total error
        return error

    def constraint1(x):
        return 0

    for k in range(pos - s, pos + s):
        if y[k] > y[pos]:
            pos = k

    x0 = np.array([amp, wid])  # master vector to optimize, initial values
    bnds = [[0, max(y)], [1, 1000]]
    con1 = {'type': 'ineq', 'fun': constraint1}
    cons = ([con1])
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
    x = solution.x

    fit = []
    for l in range(len(ax)):  # for all the points
        fit.append(x[0] * x[1] ** 2 / ((ax[l] - ax[int(pos)]) ** 2 + x[1] ** 2))
    return fit


def decbound(xx, yy, xlims, ylims, divs=0.5):
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
    cmx = list(xx)  # centroids
    cmy = list(yy)
    divs = float(divs)  # step
    map_x = list(np.array(xlims) / divs)  # mapping limits
    map_y = list(np.array(ylims) / divs)

    x_divs = int((map_x[1] - map_x[0]))  # mapping combining 'divs' & 'lims'
    y_divs = int((map_y[1] - map_y[0]))

    x_cords = [divs * i for i in range(int(min(map_x)), int(max(map_x)))]  # coordinates
    y_cords = [divs * i for i in range(int(min(map_y)), int(max(map_y)))]

    pmap = [[0 for _ in range(x_divs)] for _ in range(y_divs)]  # matrix

    for i in range(len(x_cords)):
        for j in range(len(y_cords)):
            t1 = 9999999999999999999999999
            # grad = [] #####
            for k in range(len(cmx)):
                d = (cmx[k] - x_cords[i]) ** 2 + (cmy[k] - y_cords[j]) ** 2
                # grad.append(d) ####
                if d < t1:
                    t1 = d
                    pmap[j][i] = k
            # grad = np.array(grad)/sum(grad) #####
            # pmap[j][i] = 0 ####
            # for k in range(len(grad)): ####
            #    pmap[j][i] += len(grad) - (k+1)*grad[k] ###
    return pmap


def regression(target, variable, cov=0):
    """
    Performs an N dimensional regression.

    :type target: list[float]
    :param target: Y-axis values, values to predict.

    :type variable: list[float]
    :param variable: X-axis values.

    :type cov: int
    :param cov: If 1 is regression with covariance, like spearman. The default is 0.

    :returns: Prediction of the fitting and the Fitting parameters.
    :rtype: list[list[float],list[float]]
    """
    target = list(target)
    master = list(variable)
    cov = int(cov)

    if cov == 1:
        master.append(target)  # array of arrays
        pos = [i for i in range(len(master[0]))]  # ascending values
        df = pd.DataFrame(data=master[0], columns=['0'])  # 1st col is 1st var
        for i in range(len(master)):  # for all variables and target
            df[str(2 * i)] = master[i]  # insert into dataframe
            df = df.sort_values(by=[str(2 * i)])  # sort ascending
            df[str(1 + 2 * i)] = pos  # translate to position
            df = df.sort_index()  # reorder to maintain original position

        master = [df[str(2 * i + 1)] for i in range(int(len(df.columns) / 2 - 1))]
        target = df[str(len(df.columns) - 1)]

    A = []  # fit function parameter
    for i in range(len(master[0])):  # for all the data points
        v = [1 for i in range(len(master) + 1)]
        for j in range(len(master)):
            v[j] = master[j][i]
        A.append(v)  # A value

    b = np.matrix(target).T  # transpose matrix
    A = np.matrix(A)
    fit = (A.T * A).I * A.T * b  # evaluate fir

    prediction = []
    for i in range(len(master[0])):
        p = 0
        for j in range(len(master)):
            p += master[j][i] * fit[j]
        p += fit[len(fit) - 1]
        prediction.append(float(p))

    return prediction, fit


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
    divs = float(divs)

    map_x = list(np.array(lims[:2]) / divs)  # mapping limits
    map_y = list(np.array(lims[2:]) / divs)

    x_p = list(x_points)
    y_p = list(y_points)

    groups = list(groups)
    n_groups = int(max(groups) + 1)  # number of groups

    x_divs = int((map_x[1] - map_x[0]))  # divisons for mapping
    y_divs = int((map_y[1] - map_y[0]))

    x_cords = [divs * i for i in range(int(min(map_x)), int(max(map_x)))]  # coordinates
    y_cords = [divs * i for i in range(int(min(map_y)), int(max(map_y)))]

    master = []  # to store the maps for each group

    for l in range(n_groups):
        pmap = [[0 for _ in range(x_divs)] for _ in range(y_divs)]  # individual matrices
        for i in range(len(x_cords) - 1):
            for j in range(len(y_cords) - 1):
                count = 0
                for k in range(len(x_p)):
                    if (x_cords[i] < x_p[k] <= x_cords[i+1] and
                            y_cords[j] < y_p[k] <= y_cords[j+1] and
                            groups[k] == l):
                        count += 1
                        
              
                if count > 1:  ###############    
                    pmap[j][i] = count
                else:
                    pmap[j][i] = 0  #########


        maximum = max(np.array(pmap).flatten())
        pmap = np.array(pmap) / maximum
        master.append(pmap)
    return master


def colormap(colors, name='my_name', n=100):
    """
    Just to simplify a bit the creation of colormaps.

    :type colors: list
    :param colors: List of colors thaat you ant on the colormap

    :type name: string, optional
    :param name: Name of the colormap. The default is 'my_name'.

    :type n: int, optional
    :param n: Divisions. The default is 100.

    :returns: Colormap in Matplotlib format.
    :rtype: cmap
    """    
    return LinearSegmentedColormap.from_list(name, colors, N=n)


def isaxis(data):
    """
    Detects if there is an axis in the data.

    :type data: list[float]
    :param data: Data containing spectras an possible axis.

    :returns: True if there is axis.
    :rtype: bool
    """
    features = list(data)

    is_axis = True  # there is axis by default
    x_axis = features[0]  # axis should be the first
    for i in range(len(x_axis) - 1):  # check all the vector
        if x_axis[i] > x_axis[i + 1]:
            is_axis = False
            break
    return is_axis


def trim(data, start=0, finish=0):
    """
    Deletes columns in a list from start to finish.

    :type data: list
    :param data: Data to be trimmed. Single vector at the moment.

    :type start: int, optional
    :param start: Poistion of the starting point. The default is 0.

    :type finish: int, optional
    :param finish: Position of the ending point (not included). The default is 0.

    :returns: Trimmed data.
    :rtype: list[]
    """
    data = copy.deepcopy(data)

    dims = len(np.array(data).shape)
    if dims > 1:
        final = []
        if finish == 0 or finish > len(data[0]):
            finish = len(data[0])
        t = finish - start    
        for j in data:
            temp = j
            for i in range(t):
                temp = np.delete(temp, start, 0)   
            final.append(list(temp))
        data = final
        
    else:
        if finish == 0 or finish > len(data):
            finish = len(data)
    
        t = finish - start
    
        for i in range(t):
            data = np.delete(data, start, 0)
            
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
    all_list = copy.deepcopy(arrays)
    delratio = float(delratio)

    features = all_list[0]
    for i in range(1, len(all_list)):
        features = np.c_[features, all_list[i]]

    np.random.shuffle(features)  # shuffle data before training the ML

    if delratio > 0:  # to random delete an amount of data
        if delratio >= 1:
            delratio = 0.99

        delnum = int(math.floor(delratio * len(features)))  # amount to delete

        for i in range(delnum):
            row = random.randrange(0, len(features))  # row to delete
            features = np.delete(features, row, 0)

    new_list = [[] for _ in all_list]
    lengths = []

    for i in range(len(all_list)):
        if len(np.array(all_list[i]).shape) > 1:  # if lists are 2d
            lengths.append(np.array(all_list[i]).shape[1])  # save the length
        else:
            lengths.append(1)  # otherwise is only 1 number
        
    for i in range(len(all_list)):
        start = sum(lengths[0:i])
        finish = sum(lengths[0:i+1])
        for j in range(len(features)):
            if lengths[i] == 1:
                new_list[i].append(float(features[j][start:finish]))
            else:
                new_list[i].append(features[j][start:finish])

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
    data = list(data)  # list of lists
    master = [[] for _ in data[0]]

    for i in range(len(data[0])):
        for j in range(len(data)):
            dims = len(np.array(data[j]).shape)
            if dims > 1:
                master[i].extend(data[j][i])
            else:
                master[i].extend([data[j][i]])
    return master


def shiftref(ref_data, ref_axis, ref_peak=520, mode=1, it=100, plot=True):
    """
    Shifts the x-axis according to a shift calculated prior.

    :type ref_data: list[float]
    :param ref_data: Reference measurement.

    :type ref_axis: list[float]
    :param ref_axis: X-axis of the reference measurement.

    :type ref_peak: float
    :param ref_peak: Where the reference peak should be in x-axis values. The default is 520 (Raman Si).

    :type mode: int
    :param mode: Fitting method, Lorentz, Gaussian, or none (1,2,3). The default is 1.

    :type it: int
    :param it: Fitting iterations. The default is 100.

    :type plot: bool
    :param plot: If True plots a visual aid. The default is True.

    :returns: Shift amount
    :rtype: float
    """
    ref_data = list(ref_data)
    ref_axis = list(ref_axis)
    ref_peak = float(ref_peak)
    mode = int(mode)
    it = int(it)
    plot = bool(plot)

    fit = []  # fit curves(s), if selected
    shift = []  # axis shift array
    
    dims = len(np.array(ref_data).shape)
    if dims > 1:
        for i in range(len(ref_data)):  # depending on the mode chosen...
            if mode == 1:
                fit.append(lorentzfit(ref_data[i], ref_axis[i], 4, it))
            if mode == 2:
                fit.append(gaussfit(ref_data[i], ref_axis[i], 4.4, it))
            if mode == 0:
                fit.append(ref_data[i])
    else:
        if mode == 1:
            fit.append(lorentzfit(ref_data, ref_axis, 4, it))
        if mode == 2:
            fit.append(gaussfit(ref_data, ref_axis, 4.4, it))
        if mode == 0:
            fit.append(ref_data)

    for i in range(len(fit)):  # look for the shift with max value (peak)
        for j in range(len(fit[0])):  # loop in all axis
            if fit[i][j] == max(fit[i]):  # if it is the max value,
                if dims > 1:
                    shift.append(ref_axis[i][j] - ref_peak)  # calculate the diference
                else:
                    shift.append(ref_axis[j] - ref_peak)  # calculate the diference

    temp = 0  # temporal variable
    for i in range(len(shift)):
        temp = temp + shift[i]  # make the average

    peakshift = -temp / len(shift)

    if plot:
        plt.figure()  # figsize = (16,9)
        for i in range(len(ref_data)):
            plt.plot(ref_axis[i], ref_data[i], linewidth=2, label='Original' + str(i))
            plt.plot(ref_axis[i], fit[i], linewidth=2, label='Fit' + str(i), linestyle='--')
        plt.axvline(x=ref_peak, ymin=0, ymax=max(ref_data[0]), linewidth=2, color="red", label=ref_peak)
        plt.axvline(x=ref_peak - peakshift, ymin=0, ymax=max(ref_data[0]), linewidth=2, color="yellow",
                    label="Meas. Max.")
        plt.gca().set_xlim(ref_peak - 15, ref_peak + 15)
        # plt.gca().set_ylim(0, 2)
        plt.legend(loc=0)
        plt.ylabel('a.u.')
        plt.xlabel('Shift (cm-1)')
        plt.show()

    return peakshift


def classify(data, gnumber=3, glimits=[], var='x'):
    """
    Classifies targets according to either defined limits or number of groups.
    The latter depends on the defined parameters.

    :type data: list[float]
    :param data: Vector with target values to be classified.

    :type gnumber: int
    :param gnumber: Number of groups to create. The default is 3 and is the default technique.

    :type glimits: list[float]
    :param glimits: Defined group limits. The default is [].
    
    :type var: string
    :param glimits: Name of the variable that is being classified.

    :returns: Vector with the classification from 0 to N. A list with strings with the name of the groups, useful for plotting.
    :rtype: list[list,list]
    """
    group_number = int(gnumber)
    group_limits = list(glimits)
    targets = list(data)

    # df_targets['T'] is data from the file, ['NT'] the classification (0 to ...)
    df_targets = pd.DataFrame(data=targets, columns=['T'])
    group_names = []
    class_targets = []  # ['NT']

    # if I set the number of groups
    if group_number > 0:
        group_limits = [[0, 0] for _ in range(group_number)]
        df_targets.sort_values(by='T', inplace=True)
        group_size = math.floor(len(targets) / group_number)
        left_over = len(targets) - group_size * group_number
        g_s = []
        for i in range(group_number):
            g_s.append(group_size + math.floor((i + 1) / group_number) * left_over)
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
            else:
                group_limits[i][1] = df_targets['T'].iloc[int(temp + g_s[i])]
            temp = temp + g_s[i]   
        
        group_names.append(str(var)+' < ' + str(group_limits[0][1]))
        for i in range(0, len(group_limits) - 2):
            group_names.append(str(group_limits[i][1]) + ' <= '+str(var)+' < ' + str(group_limits[i + 1][1]))
        group_names.append(str(group_limits[len(group_limits)-1][0]) + ' <= '+str(var))
        
        df_targets.sort_index(inplace=True)
        class_targets = list(df_targets['NT'])
        
        # if I set the limits
    if len(group_limits) >= 1 and group_number <= 1:
        class_targets = [-1 for _ in range(len(targets))]

        for i in range(0, len(group_limits)):
            for j in range(len(targets)):

                if targets[j] < group_limits[0]:
                    class_targets[j] = 0

                if targets[j] >= group_limits[len(group_limits) - 1]:
                    class_targets[j] = len(group_limits)
                # targets[j] >= group_limits[i] and targets[j] < group_limits[i+1]:
                elif group_limits[i] <= targets[j] < group_limits[i + 1]:
                    class_targets[j] = i + 1

        group_names.append(str(var)+' < ' + str(group_limits[0]))
        for i in range(0, len(group_limits) - 1):
            group_names.append(str(group_limits[i]) + ' <= '+str(var)+' < ' + str(group_limits[i + 1]))
        group_names.append(str(max(group_limits)) + ' <= '+str(var))

    return class_targets, group_names


def subtractref(data, ref, axis=0, alpha=0.9, sample=0, plot_lim=[0, 0], plot=False):
    """
    Subtracts a reference spectra (i.e.: air) from the measurements.

    :type data: list[float]
    :param data: List of or single vector.

    :type ref: list[float]
    :param ref: reference data to remove.

    :type axis: list[float]
    :param axis: Axis for both ´data´ and 'ref', only for plotting purposes.

    :type alpha: float
    :param alpha: Manual multiplier. The default is 0.9.

    :type sample: int
    :param sample: Sample spectra to work with. The default is 0.

    :type plot_lim: list[int]
    :param plot_lim: Limits of the plot.

    :type plot: bool
    :param plot: To plot or not a visual aid. The default is True.

    :returns: Data with the subtracted reference.
    :rtype: list`[float]
    """
    data = copy.deepcopy(data)
    ref = list(ref)
    sample = int(sample)  # spectrum chosen to work with. 0 is the first
    alpha = float(alpha)  # multiplication factor to delete the ref spectrum on the sample
    
    dims = len(np.array(data).shape)
    
    if dims > 1:
        for i in range(len(data)):
            data[i] = np.array(data[i]) - np.array(ref) * alpha  
        toplot = data[sample]
        final =  np.array(toplot) -  np.array(ref) * alpha
    else:
        toplot = copy.deepcopy(data)
        final = np.array(data) - np.array(ref) * alpha    
        data = final
        
    if plot:
        if axis == 0:
                axis = [i for i in range(len(ref))]
        else:
            axis = list(axis)
        
        plt.plot(axis, toplot, linewidth=1, label='Original', linestyle='--')
        plt.plot(axis, ref, linewidth=1, label='Air', linestyle='--')
        plt.plot(axis, final, linewidth=1, label='Final')
        if plot_lim[0] < plot_lim[1]:
            plt.gca().set_xlim(plot_lim[0], plot_lim[1])
        plt.legend(loc=0)
        plt.ylabel('a.u.')
        plt.xlabel('Shift (cm-1)')
        plt.show()

    return list(data)


def pearson(data, labels=[], cm="seismic", fons=20, figs=(20, 17), tfs=25, 
            ti="Pearson", plot=True):
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

    :type plot: bool
    :param plot: If True plots the matrix. The default is True.

    :type ti: str
    :param ti: Plot title/name. The default is "spearman".
    """
    labels = list(labels)
    cm = str(cm)
    fonsize = float(fons)
    figsize = list(figs)
    titlefs = float(tfs)
    title = str(ti)
    n = len(data)

    if len(labels) < 1:
        labels = [i for i in range(len(data))]

    pears = []
    cordsi = []  # coordinates, same order as labels
    cordsj = []  # coordinates, same order as labels
    for i in range(n):  # for all sets
        for j in range(n):  # compare with all sets
            x = data[i]
            y = data[j]
            pears.append(stats.pearsonr(x, y)[0])

            cordsi.append(int(i))
            cordsj.append(int(j))

    merged_pears = np.c_[pears, cordsi, cordsj]  # merge to sort together
    merged_pears = sorted(merged_pears, key=lambda l: l[0], reverse=True)

    for i in range(n):  # delete the first n (the obvious 1s)
        merged_pears = np.delete(merged_pears, 0, 0)

    for i in range(int((n * n - n) / 2)):  # delete the repeated half
        merged_pears = np.delete(merged_pears, i, 0)

    pears = np.reshape(pears, (n, n))  # [pearson coeff, p-value]

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    plt.rc('font', size=fonsize)
    y = [i + 0.5 for i in range(n)]
    ticks = mpl.ticker.FixedLocator(y)
    formatt = mpl.ticker.FixedFormatter(labels)
    if plot:
        fig = plt.figure(tight_layout=True, figsize=figsize)    
        ax = fig.add_subplot(gs[0, 0])
        pcm = ax.pcolormesh(pears, cmap=cm, vmin=-1, vmax=1)
        fig.colorbar(pcm, ax=ax)
        ax.set_title(title, fontsize=titlefs)
        ax.xaxis.set_major_locator(ticks)
        ax.yaxis.set_major_locator(ticks)
        ax.xaxis.set_major_formatter(formatt)
        ax.yaxis.set_major_formatter(formatt)
        plt.xticks(rotation='90')
        plt.show()

    return pears

def spearman(data, labels=[], cm="seismic", fons=20, figs=(20, 17), tfs=25, ti="Spearman", plot=True):
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

    :type plot: bool
    :param plot: If True plots the matrix. The default is True.

    :type ti: str
    :param ti: Plot title/name. The default is "spearman".
    """
    labels = list(labels)
    cm = str(cm)
    fonsize = float(fons)
    figsize = list(figs)
    titlefs = float(tfs)
    title = str(ti)
    n = len(data)

    if len(labels) < 1:
        labels = [i for i in range(len(data))]

    spear = []  # spearman
    cordsi = []  # coordinates, same order as labels
    cordsj = []  # coordinates, same order as labels
    for i in range(n):
        for j in range(n):  # compare with all sets
            x = data[i]
            y = data[j]
            spear.append(stats.spearmanr(x, y)[0])

            cordsi.append(int(i))
            cordsj.append(int(j))

    merged_spear = np.c_[spear, cordsi, cordsj]  # merge to sort together
    merged_spear = sorted(merged_spear, key=lambda l: l[0], reverse=True)

    for i in range(n):  # delete the first n (the obvious 1s)
        merged_spear = np.delete(merged_spear, 0, 0)

    for i in range(int((n * n - n) / 2)):  # delete the repeated half
        merged_spear = np.delete(merged_spear, i, 0)

    spear = np.reshape(spear, (n, n))  # [rho spearman, p-value]

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    plt.rc('font', size=fonsize)
    y = [i + 0.5 for i in range(n)]
    ticks = mpl.ticker.FixedLocator(y)
    formatt = mpl.ticker.FixedFormatter(labels)
    if plot:
        fig = plt.figure(tight_layout=True, figsize=figsize)
        ax = fig.add_subplot(gs[0, 0])
        pcm = ax.pcolormesh(spear, cmap=cm, vmin=-1, vmax=1)
        fig.colorbar(pcm, ax=ax)
        ax.set_title(title, fontsize=titlefs)
        ax.xaxis.set_major_locator(ticks)
        ax.yaxis.set_major_locator(ticks)
        ax.xaxis.set_major_formatter(formatt)
        ax.yaxis.set_major_formatter(formatt)
        plt.xticks(rotation='90')
        plt.show()
    
    return spear


def grau(data, labels=[], cm="seismic", fons=20, figs=(25, 15),
         tfs=25, ti="Grau (Beta)", marker="s", marks=100, plot=True):
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

    :type plot: bool
    :param plot: If True plots the matrix. The default is True.

    :type marks: int
    :param marks: DESCRIPTION. The default is 100.
    """
    labels = list(labels)
    cm = str(cm)
    fontsize = float(fons)  # plot font size
    figsize = list(figs)  # plot size
    titlefs = float(tfs)  # title font size
    title = str(ti)  # plot name
    marker = str(marker)  # market style
    markersize = float(marks)  # marker size

    if len(labels) < 1:
        labels = [i for i in range(len(data))]

    graus = evalgrau(data)  # grau correlation (3d R2)
    g1 = [graus[i][0] for i in range(len(graus))]  # first dimension values
    g2 = [graus[i][1] for i in range(len(graus))]  # second dimension values
    g3 = [graus[i][2] for i in range(len(graus))]  # third dimension values
    mse = [graus[i][3] for i in range(len(graus))]  # mse's list
    g2_shift = list(g2)  # copy v2 to displace the d2 values for plotting
    t_c = []  # list of ticks per subplot
    xtick_labels = []  # list for labels
    c = 1  # number of different # in combs[i][0] (first column), starts with 1

    for i in range(len(graus) - 1):  # for all combinations
        if graus[i][0] != graus[i + 1][0]:  # if it is different
            c += 1  # then it is a new one

    for i in range(c):  # for all the different first positions
        temp = 0  # temporal variable to count ticks
        for j in range(len(graus) - 1):  # check all the combinations
            if graus[j][0] == i:  # if it is the one we are looking for
                if graus[j][1] != graus[j + 1][1]:  # if it changes, then new tick
                    temp += 1  # add
        if temp == 0:  # if it doesnt count, is because there is only 1
            t_c.append(1)  # so append 1
        elif temp == 1:  # if only 1 is counted
            t_c.append(2)  # then add a 2
        else:  # otherwise,
            t_c.append(temp)  # when it is done, append to the list

    for i in range(len(t_c)):  # for all the different #
        for j in range(t_c[i] + 1):  # for the number of different 2d
            xtick_labels.append(j + i)  # append the value +1
    xtick_labels.append('')  # append '' for the last label to be blank

    for i in range(len(g1)):  # to shift the position for plotting
        for j in range(1, len(t_c)):  # for all the ticks in x axis
            if g1[i] >= j:  # if bigger than 0 (first)
                g2_shift[i] += t_c[j - 1]  # shift (add) all the previous


    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    plt.rc('font', size=fontsize)   
    cm = plt.cm.get_cmap(cm)
    y_ticks = [i for i in range(int(min(g3)), int(max(g3)) + 1)]
    x_ticks = [i for i in range(int(max(g2_shift)) + 2)]
    ytick_labels = []
    for i in range(len(y_ticks)):
        ytick_labels.append(labels[y_ticks[i]])
    for i in range(len(xtick_labels) - 1):
        xtick_labels[i] = labels[xtick_labels[i]]
    if plot:
        fig = plt.figure(tight_layout=True, figsize=figsize)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title(title, fontsize=titlefs)
        sc = ax.scatter(g2_shift, g3, alpha=1, edgecolors='none',
                        c=mse, cmap=cm, s=markersize, marker=marker)
        plt.colorbar(sc)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xlim(0, max(x_ticks))
        ax.get_xticklabels()[0].set_fontweight('bold')
        plt.xticks(rotation='90')
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
    
    return g2_shift


def moveavg(data, move=2):
    """
    Calculate the moving average of a single or multiple vectors.

    :type data: list[float]
    :param data: Data to calculate the moving average. Single or multiple vectors.

    :type move: int
    :param move: Average range to each side (total average = move + 1).

    :returns: Smoothed vector(s).
    :rtype: list[float]
    """
    move = int(move)
    b_data = copy.deepcopy(data)
    dims = len(np.array(b_data).shape)
    avg = []  # for smoothed data

    if dims > 1:
        data_len = len(b_data[0])

        for j in range(len(b_data)):  # each measured point
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


def plot2dml(train, test=[], names=['D1', 'D2', 'T'], train_pred=[], 
             test_pred=[], labels=[],title='', xax='x', yax='y', fs=15, 
             lfs=10, loc='best', size=20, xlim=[], ylim=[], plot=True):
    """
    Plots 2-dimensional results from LDA, PCA, NCA, or similar machine learning
    algoruthms where the output has 2 features per sample.
    
    :type train: pandas frame
    :param train: Results for the training set. Pandas frame with the 2 dimensions
        and target columns.

    :type test: pandas frame
    :param test: Results for the test set. Pandas frame with the 2 dimensions
        and target columns.
    
    :type names: list[str]
    :param names: Name of the lables in the dataframe. For example, for LDA:
        D1, D2 and T.    
    
    :type train_pred: list
    :param train_pred: Prediction of the training set.

    :type test_pred: list
    :param test_pred: Prediction of the test set.

    :type labels: list
    :param labels: Names for the classification groups, if any.
        
    :type title: str
    :param title: Title of the plot.
        
    :type xax: str
    :param xax: Name ox x-axis
        
    :type yax: str
    :param yax: Name of y-axis
        
    :type lfs: int
    :param lfs: Legend font size. Default is 15.
        
    :type loc: str
    :param loc: Location of legend. Default is best.
    
    :type size: int
    :param size: Size of the markers. Default is 20.
    
    :type xlim: list
    :param xlim: Limits of the x axis.
    
    :type ylim: list
    :param ylim: Limits of the y axis.
    
    :type plot: bool
    :param plot: If True it plot. Only for test purposes.
        
    :returns: Plot
    """
    marker = ['o', 'v', 's', 'd', '*', '^', 'x', '+', '.',
              'o', 'v', 's', 'd', '*', '^', 'x', '+', '.']
    color = ['orange', 'green', 'blue', 'grey', 'yellow', 'olive', 'lime',
             'springgreen', 'mediumspringgreen', 'cyan', 'teal', 'royalblue', 
             'turquoise', 'indigo', 'purple', 'deeppink', 'crimson']
    
    legend_elements = []
    if len(labels) > 0:
        
        for j in range(int(max(train['T'])+1)):
            legend_elements.append(Line2D([0], [0], marker=marker[j], color='w', 
                                   label=labels[j], markerfacecolor=color[j], markersize=5)) 
        
    for i in range(len(train)):  
        group = int(train['T'][i])
        ec = 'none'
        
        if len(train_pred) > 1:
            if train_pred[i] != train[names[2]][i]:
                ec = 'red'
        if plot:
            plt.scatter(train[names[0]][i], train[names[1]][i], alpha=0.7, s=size,
                        linewidths=1, color=color[group], marker=marker[group],
                        edgecolor=ec)
            
    for i in range(len(test)):        
        group = int(test['T'][i])
        ec = 'black'
        
        
        if len(test_pred) > 1:
            if test_pred[i] != test[names[2]][i]:
                ec = 'fuchsia'
                
        if plot:
            plt.scatter(test[names[0]][i], test[names[1]][i], alpha=0.5, s=size,
                        linewidths=1, color=color[group], marker=marker[group],
                        edgecolor=ec)
            
    if plot:
        plt.rc('font', size=fs)
        plt.xlabel(xax)
        plt.ylabel(yax)
        plt.title(title)
        if len(xlim) > 0:
            plt.xlim(xlim[0],xlim[1])
        if len(ylim) > 0:    
            plt.ylim(ylim[0],ylim[1])
        plt.legend(handles=legend_elements, loc=loc, fontsize=lfs)
        plt.show()
    
    return plot

    
def stackplot(data, add, xlabel='', ylabel='', cmap='Spectral', 
              figsize=(3, 4.5), lw=1, plot=True):
    """
    Plots a stack plot of selected spectras.

    :type data: list[float]
    :param data: Data to in the plot.

    :type add: float
    :param add: displacement, or difference, between each curve.

    :type xlabel: str
    :param xlabel: Label of axis.

    :type ylabel: str
    :param ylabel: Label of axis.

    :type cmap: str
    :param cmap: Colormap, according to matplotlib options.

    :type figsize: tuple
    :param figsize: Size of the plot. Default is (3, 4.5)

    :type lw: float
    :param lw: Linewidth of the curves.

    :type plot: bool
    :param plot: If True it plot. Only for test purposes.

    :returns: plot
    :rtype: bool
    """      
       
    base = [add for _ in range(len(data[0]))]
    
    cmap = plt.cm.get_cmap(cmap)
    color = []
    for i in range(len(data)):
        color.append(cmap(i/(len(data)-1)))
    
    if plot:
        plt.figure(figsize=figsize)
        for i in range(len(data)):
            plt.plot(np.array(data[i]) + np.array(base)*i, color=color[i], lw=lw)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()  
        
    return plot


def cosmicmp(ndata, alpha=1, avg=2):
    """
    https://doi.org/10.1177/0003702819839098
    It identifies CRs by comparing similar spectras and paring in matching
    pairs. Uses randomnes of CRs.
    
    :type ndata: list[float]
    :param ndata: List of spectras to remove cosmic rays.

    :type alpha: float
    :param alpha: Factor to modify the criteria to identify a cosmic ray.

    :returns: Data with removed cosmic rays.
    :rtype: list[float]
    """
    data = copy.deepcopy(ndata)

    sim_specs = []  # most similar spectra (position) for each spectra i

    for i in range(len(data)):  # for all the spectras
        n_cov = -1  # position of most similar spectra, initial value
        cov = 0  # to compare similarity of the spectras
        paired = 0  # if it is already paired with another spectra
        for j in range(len(sim_specs)):  # check if it is already paired
            if sim_specs[j] == i:  # if it is
                n_cov = j  # claculate nothing and just save it
                paired = 1  # set as paired
        
        if paired == 0:  # if not paired, then calculate
            b = np.dot(data[i], data[i])  # first term of equation
            for j in range(len(data)):  # search in all spectras
                a = np.dot(data[i], data[j])**2  # second term of eq.
                c = np.dot(data[j], data[j])  # third term of eq.
                temp = a/(b*c)  # final value to campare
                if temp > cov and j != i:  # the highest value (covariance) wins
                    n_cov = j  # save the best
                    cov = temp  # save the best to compare
        
        sim_specs.append(n_cov)  # save the similar curve
        
        mavg = moveavg(data[i], avg)
        
        sigma = 0
        for j in range(len(data[i])):
            sigma += math.sqrt((data[i][j] - mavg[j])**2)
        sigma = sigma / len(data[i])

        for j in range(len(data[0])):  # search in all the spectra
            if data[i][j] - data[sim_specs[i]][j] > sigma*alpha:  # if res. is higher than the stdev
                data[i][j] = data[sim_specs[i]][j]  # must be CR, change the value
                        
    return data


def cosmicdd(ndata, th=100, asy=0.6745, m=5):
    """
    https://doi.org/10.1016/j.chemolab.2018.06.009
    It identifies CRs by detrended differences, the differences between a
    value and the next.
    
    :type ndata: list[float]
    :param ndata: List of spectras to remove cosmic rays.

    :type th: float
    :param th: Factor to modify the criteria to identify a cosmic ray.
    
    :type asy: float
    :param asy: Asymptotic bias correction
    
    :type m: float
    :param m: Number of neighbor values to use for average.

    :returns: Data with removed cosmic rays.
    :rtype: list[float]
    """
    data = copy.deepcopy(ndata)
    
    diff = list(np.array(data))  # diff data
     
    for i in range(len(data)):  # for each spectra
        for j in range(len(data[0])-1):  # for each step
            diff[i][j] = abs(data[i][j]-data[i][j+1])  # diff with the next one
    
    zt = []  # Z scores
    for i in diff:  # for each diff. vector
        z = []  # temporal z score
        temp = []  # temporal MAD (median absolute deviation)
        med = np.median(i)  # just median
        
        for j in i:  # for each step in each diff. spectra
            temp.append(abs(j-med))  # calculate MAD
        mad = np.median(temp)  # save MAD
        
        for j in i:  # for each step in each diff. spectra
            z.append(asy*(j - med)/mad)  # calculate Z score
        zt.append(z)  # save Z score
    
    for i in range(len(data)):  # for each spectra
        for j in range(len(data[i])-1):  # in all its len. except the last (range)
            if abs(zt[i][j]) > th:  # if it is larger than the th. then it is CR
                data[i][j] = (sum(data[i][j-m:j]) + sum(data[i][j+1:j+m+1]))/(2*m)  # avg, of neighbors
    
    return data


def cosmicmed(data, sigma=1.5):
    """
    Precise cosmic ray elimination for measurements of the same point or very
    similar spectras.
    
    :type data: list[float]
    :param data: List of spectras to remove cosmic rays.

    :type sigma: float
    :param sigma: Factor to modify the criteria to identify a cosmic ray.
        Multiplies the median of each bin.

    :returns: Data with removed cosmic rays.
    :rtype: list[float]
    """
    solved = copy.deepcopy(data)     
    acq = len(solved)
    length = len(solved[0])
    
    med = median(solved)
    
    for i in range(acq):
        for j in range(length):
            if data[i][j] > sigma*med[j]:
                solved[i][j] = med[j]
    
    return solved


def makeaxisstep(start=0, step=1.00, length=1000, adjust=False, rounded=-1):
    """
    Creates an axis, or vector, from 'start' with bins of length 'step' 
    for a distance of 'length'.
    
    :type start: float 
    :param start: first value of the axis. Default is 0.
        
    :type step: float
    :param step: Step size for the axis. Default is 1.00.
        
    :type length: int
    :param length: LEngth of axis. Default is 1000.
    
    :type adjust: boolean
    :param adjust: If True, rounds (adjusts) the deimals points to the same
        as the step has. Default is False.
        
    :type rounded: int 
    :param rounded: Number of decimals to consider. If -1 then no rounding is
        performed.
        
    :returns: Axis with the set parameters.
    :rtype: list[float]
    """
    if adjust:
        d = str(step)[::-1].find('.')
        axis = [round(start+step*i,d) for i in range(length)]
    else:
        if rounded >= 0:
            axis = [round(start+step*i,rounded) for i in range(length)]
        else:
            axis = [start+step*i for i in range(length)]
    return axis


def makeaxisdivs(start, finish, divs, rounded=-1):
    """
    Creates an axis, or vector, from 'start' to 'finish' with 'divs' divisions. 
    
    :type start: float 
    :param start: First value of the axis.
        
    :type finish: float
    :param finish: Last value of the axis.
        
    :type divs: int
    :param divs: Number of divisions
        
    :type rounded: int 
    :param rounded: Number of decimals to consider. If -1 then no rounding is
        performed.
        
    :returns: Axis with the set parameters.
    :rtype: list[float]
    """
    step = (finish-start)/(divs-1)
    if rounded >= 0:
        axis = [round(start+step*i,rounded) for i in range(divs)]
    else:
        axis = [start+step*i for i in range(divs)]
    return axis


def minmax(data):
    """
    Calculates the vectors that contain the minimum and maximum values of each
    bin from a list of vectors.
    
    :type data: list 
    :param data: List of vectors to calculate the minimum and maximum vectors.
                
    :returns: minimum and maximum vectors.
    :rtype: list[float]
    """
    minimum = []
    maximum = []
    for i in range(len(data[0])):
        temp_min = 99999999999999999999
        temp_max = -9999999999999999999
        for j in range(len(data)):
            if data[j][i] < temp_min:
                temp_min = data[j][i]
            if data[j][i] > temp_max:
                temp_max = data[j][i] 
        minimum.append(temp_min)
        maximum.append(temp_max)
    return minimum, maximum
