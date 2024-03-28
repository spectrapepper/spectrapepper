"""
This main module contains all the functions available in spectrapepper. Please
use the search function to look up specific functionalities and keywords.
"""

from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import splev, splrep
from scipy.signal import butter, filtfilt
from scipy.sparse.linalg import spsolve
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy import interpolate
from scipy.stats import stats
import statistics as sta
import matplotlib as mpl
import statistics as sta
from scipy import sparse
import pandas as pd
import numpy as np
import linecache
import itertools
import os.path
import random
import math
import copy
import os


def load_spectras(sample=None):
    """
    Load sample specrtal data, axis included in the first line.

    :type sample: int, tuple
    :param sample: Index of the sample spectra wanted or a tuple with the range
        of the indeces of a groups of spectras.

    :returns: X axis and the sample spectral data selected.
    :rtype: list[float], list[float]
    """
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'spectras.txt')
    data = load(my_file)
    x = data[0]
    y = data[1:]

    if isinstance(sample, tuple):
        y = y[sample[0]: sample[1]]
    elif isinstance(sample, int):
        y = y[sample]

    return x, y


def load_targets(flatten=True):
    """
    Load sample targets data for the spectras.

    :returns: Sample targets.
    :rtype: list[float]
    """

    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'targets.txt')
    data = load(my_file)

    if flatten:
        data = np.array(data).flatten()

    return data


def load_params(transpose=True):
    """
    Load sample parameters data for the spectras.

    :returns: Sample parameters.
    :rtype: list[float]
    """

    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'params.txt')
    data = load(my_file)

    if transpose:
        data = np.transpose(data)

    return data


def load(file, format="row", fromspectra=0, tospectra=None, spectra=None,
         transpose=False, dtype=float, separators=[";"],
         blanks=["NaN", "nan", "--"], replacewith="0"):
    """
    Load data from a text file ('.txt') obtained from LabSpec and other
    spectroscopy software, or from an Excel file ('.xlsx', '.xls'). Normally,
    single measurements come in columns with the first one being the x-axis.
    When it is a mapping, the first row is the x-axis and the following are
    the measurements. It can also perform random access to a particular
    ´spectra´ and also it is possible to define specific range to load with
    'fromspectra' and 'tospectra'.

    :type file: str
    :param file: Url of data file. Must not have headers and separated by
        'spaces' (LabSpec).

    :type format: str
    :param format: Format of the spectra in the file. It can be "row" or
        "column". Generally, it will be "column" for Excel files. The default
        is "row".

    :type fromspectra: int
    :param fromspectra: Spectra of file from which to start loading data. The
        default is 0.

    :type tospectra: int
    :param tospectra: Spectra of file to which to end loading data. The default
        is 'None' which idicates the full range of data.

    :type spectra: int
    :param spectra: Random access to file. Loads a specific spectra in a file.
        Default is ´None´.

    :type transpose: boolean
    :param transpose: If True transposes the data. Default is False.
    :type dtype: str
    :param dtype: Type of data. If its numeric then 'float', if text then
        'string'. Default is 'float'.

    :returns: List of the data.
    :rtype: list[float]
    """

    new_data = []
    def process_row(row, dtype):
        for i in separators:
            row = row.replace(i, " ")
        for i in blanks:
            row = row.replace(i, replacewith)

        row = str.split(row)
        return np.array(row, dtype=dtype)

    extension = file.split('.')[-1].lower()
    if extension == 'txt':
        if format == "row":
            with open(file, 'r') as raw_data:
                if spectra is not None:
                    new_data = linecache.getline(file, spectra + 1)
                    new_data = process_row(new_data, dtype)
                else:
                    new_data = []
                    for i, row in enumerate(raw_data):
                        if i >= fromspectra:
                            if tospectra is not None and i > tospectra:
                                break
                            s_row = process_row(row, dtype)
                            new_data.append(s_row)

                if transpose:
                    new_data = np.transpose(new_data)

        else:
            if spectra is not None:
                new_data = pd.read_csv(file, delimiter='\t', header=None,
                                       usecols=[spectra])
            elif tospectra is not None:
                new_data = pd.read_csv(file, delimiter='\t', header=None,
                                       usecols=range(fromspectra, tospectra+1))
            else:
                new_data = pd.read_csv(file, delimiter='\t', header=None)

            for i in blanks:
                if i == 'NaN' or i == 'nan':
                    new_data = new_data.fillna(float(replacewith))
                else:
                    new_data = new_data.replace(i, float(replacewith))

            if transpose:
                new_data = np.transpose(new_data)

            new_data = [np.array(new_data.iloc[:, i])
                        for i in range(0, new_data.shape[1])]

    elif extension == 'xlsx' or extension == 'xls':
        if format == "row":
            if spectra is not None:
                new_data = pd.read_excel(file, skiprows=range(0, spectra),
                                         nrows=1)
            elif tospectra is not None:
                new_data = pd.read_excel(file, skiprows=range(0, fromspectra),
                                         nrows=tospectra-fromspectra+1)
            else:
                new_data = pd.read_excel(file)
        else:
            if spectra is not None:
                new_data = pd.read_excel(file, usecols=[spectra])
            elif tospectra is not None:
                new_data = pd.read_excel(file,
                                         usecols=range(fromspectra,
                                                       tospectra+1))
            else:
                new_data = pd.read_excel(file)

        for i in blanks:
            if i == 'NaN' or i == 'nan':
                new_data = new_data.fillna(float(replacewith))
            else:
                new_data = new_data.replace(i, float(replacewith))

        if transpose:
            new_data = np.transpose(new_data)

        new_data = [np.array(new_data.iloc[:, i])
                    for i in range(0, new_data.shape[1])]

    else:
        raise TypeError(f"Error: The file extension '{extension}' is "
                        f"not recognized. "
                        f"Allowed extensions are: '.txt', '.xlsx', '.xls'")

    return new_data


def lowpass(y, cutoff=0.25, fs=30, order=2, nyq=0.75):
    """
    Butter low pass filter for a single or spectra or a list of them.

    :type y: list[float]
    :param y: List of vectors in line format (each line is a vector).

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
    y = copy.deepcopy(y)  # so it does not change the input list
    dims = len(np.array(y).shape)

    if dims == 1:
        y = [y]

    normal_cutoff = cutoff / (nyq * fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    for i in range(len(y)):
        y[i] = filtfilt(b, a, y[i])

    if dims == 1:
        y = y[0]

    return y


def normtomax(y, to=1, zeromin=False):
    """
    Normalizes spectra to the maximum value of each, in other words, the
    maximum value of each spectra is set to the specified value.

    :type y: list[float], numpy.ndarray
    :param y: Single or multiple vectors to normalize.

    :type to: float
    :param to: value to which normalize to. Default is 1.

    :type zeromin: boolean
    :param zeromin: If `True`, the minimum value is translated to 0. Default
        value is `False`

    :returns: Normalized data.
    :rtype: list[float], numpy.ndarray
    """
    y = copy.deepcopy(y)

    dims = len(np.array(y).shape)
    if dims == 1:
        y = [y]

    for i in range(len(y)):
        if zeromin:
            y[i] = y[i] - np.min(y[i])
        y[i] = y[i] / np.max(y[i]) * to

    if dims == 1:
        y = y[0]

    return y


def normtovalue(y, val):
    """
    Normalizes the spectras to a set value, in other words, the defined value
    will be reescaled to 1 in all the spectras.

    :type y: list[float]
    :param y: Single or multiple vectors to normalize.

    :type val: float
    :param val: Value to normalize to.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    y = np.array(y)/val
    y = list(y)
    return y


def alsbaseline(y, lam=100, p=0.001, niter=10, remove=True):
    """
    Calculation of the baseline using Asymmetric Least Squares Smoothing. This
    script only makes the calculation but it does not remove it. Original idea of
    this algorithm by P. Eilers and H. Boelens (2005):

    :type y: list[float]
    :param y: Spectra to calculate the baseline from.

    :type lam: int
    :param lam: Lambda, smoothness. The default is 100.

    :type p: float
    :param p: Asymmetry. The default is 0.001.

    :type niter: int
    :param niter: Niter. The default is 10.

    :type remove: True
    :param remove: If `True`, calculates and returns `data - baseline`. If
        `False`, then it returns the `baseline`.

    :returns: Returns the calculated baseline.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)

    if dims == 1:
        y = [y]

    l = len(y[0])
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(l, l - 2))
    w = np.ones(l)

    resolve = []
    for i in range(len(y)):
        for _ in range(niter):
            W = sparse.spdiags(w, 0, l, l)
            Z = W + lam * d.dot(d.transpose())
            z = spsolve(Z, w * y[i])
            w = p * (y[i] > z) + (1 - p) * (y[i] < z)
        # y[i] = y[i] - z
        if remove:
            resolve.append(y[i] - z)
        else:
            resolve.append(z)

    if dims == 1:
        resolve = resolve[0]

    return resolve


def bspbaseline(y, x, points, avg=5, remove=True, plot=False, plot_ind=0):
    """
    Calcuates the baseline using b-spline.

    :type y: list[float]
    :param y: Single or several spectra to remove the baseline from.

    :type x: list[float]
    :param x: x axis of the data, to interpolate the baseline function.

    :type points: list[float], list[[float, float]]
    :param points: Axis values of points to calculate the bspline. Axis ranges
        are also acepted. In this case, the `avg` value will be `0`.

    :type avg: int or list[int]
    :param avg: Points to each side to make average. It can be the same value
        for all points or a different value for each point. Default is 5.
        If `points` are axis ranges, then it is set to 0 and will not have
        any effect.

    :type remove: True
    :param remove: If `True`, calculates and returns `data - baseline`. If
        `False`, then it returns the `baseline`.

    :type plot: bool
    :param plot: If True, calculates and returns (data - baseline).

    :type plot_ind: int
    :param plot_ind: Index corresponding to the spectrum to be plotted.

    :returns: The baseline.
    :rtype: list[float]
    """
    data = copy.deepcopy(y)
    x = list(x)
    points = list(points)
    pos = valtoind(points, x)
    dims = len(np.array(data).shape)

    if len(np.array(points).shape) == 2:
        avg = 0
    else:
        pos = [[i, i] for i in pos]
        points = [[i, i] for i in points]

    # Convert 'avg' to a list if it wasn't, and raise an error if it doesn't
    # have the correct length.
    if not isinstance(avg, list):
        avg = [avg] * len(points)
    elif len(avg) != len(points):
        raise ValueError("The length of the 'avg' list has to be equal to "
                         "the length of the 'points' list")

    if dims == 1:
        data = [data]

    # Calculations for the 'plot_ind' index are performed first, allowing
    # quick review for potential adjustments and avoiding potential extended
    # computational time
    if plot:
        y_p = []
        for i in range(len(pos)):
            temp = np.mean(data[plot_ind][pos[i][0] - avg[i]:
                                          pos[i][1] + avg[i] + 1])
            y_p.append(temp)

        points = [np.mean(i) for i in points]
        spl = splrep(points, y_p)
        baseline_plot_ind = splev(x, spl)

        if len(np.array(baseline_plot_ind).shape) == 2:
            # sometimes this happends when doing preprocesing
            baseline_plot_ind = [h[0] for h in baseline_plot_ind]

        if remove:
            result_plot_ind = (np.array(data[plot_ind]) -
                               np.array(baseline_plot_ind))
        else:
            result_plot_ind = copy.deepcopy(baseline_plot_ind)

        plt.plot(x, data[plot_ind], label='Original')
        plt.plot(x, baseline_plot_ind, label='Baseline')
        plt.plot(points, y_p, 'o', color='red')
        plt.ylim(min(data[plot_ind]), max(data[plot_ind]))
        plt.legend()
        plt.show()

    # Calculations for the rest of spectra
    baseline = []
    result = []
    for j in range(len(data)):
        if plot and j == plot_ind:
            baseline.append(baseline_plot_ind)
            result.append(result_plot_ind)
        else:
            y_p = []
            for i in range(len(pos)):
                temp = np.mean(data[j][pos[i][0] - avg[i]:
                                       pos[i][1] + avg[i] + 1])
                y_p.append(temp)

            points = [np.mean(i) for i in points]
            spl = splrep(points, y_p)
            baseline.append(splev(x, spl))

            if len(np.array(baseline[j]).shape) == 2:
                # sometimes this happends when doing preprocesing
                baseline[j] = [h[0] for h in baseline[j]]

            if remove:
                result.append(np.array(data[j]) - np.array(baseline[j]))
            else:
                result.append(baseline[j])

    if dims == 1:
        result = result[0]

    return result


def polybaseline(y, axis, points, deg=2, avg=5, remove=True, plot=False,
                 plot_ind=0):
    """
    Calcuates the baseline using polynomial fit.

    :type y: list[float]
    :param y: Single or several spectras to remove the baseline from.

    :type axis: list[float]
    :param axis: x axis of the data, to interpolate the baseline function.

    :type points: list[int]
    :param points: positions in axis of points to calculate baseline.

    :type deg: int
    :param deg: Polynomial degree of the fit.

    :type avg: int or list[int]
    :param avg: Points to each side to make average. It can be the same value
        for all points or a different value for each point. Default is 5.
        If `points` are axis ranges, then it is set to 0 and will not have
        any effect.

    :type remove: True
    :param remove: if True, calculates and returns (y - baseline).

    :type plot: bool
    :param plot: if True, calculates and returns (y - baseline).

    :type plot_ind: int
    :param plot_ind: Index corresponding to the spectrum to be plotted.

    :returns: The baseline.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    axis = list(axis)
    points = list(points)
    pos = valtoind(points, axis)
    # Convert 'avg' to a list if it wasn't, and raise an error if it doesn't
    # have the correct length.
    if not isinstance(avg, list):
        avg = [avg] * len(points)
    elif len(avg) != len(points):
        raise ValueError("The length of the 'avg' list has to be equal to "
                         "the length of the 'points' list")

    if dims == 1:
        y = [y]

    # Calculations for the 'plot_ind' index are performed first, allowing
    # quick review for potential adjustments and avoiding potential extended
    # computational time

    if plot:
        averages = []
        for i in range(len(pos)):
            averages.append(np.mean(y[plot_ind][pos[i] - avg[i]:
                                                pos[i] + avg[i] + 1]))

        z = np.polyfit(points, averages, deg)  # polinomial fit
        f = np.poly1d(z)  # 1d polinomial
        fit = f(axis)
        if remove:
            baseline_plot_ind = y[plot_ind] - fit
        else:
            baseline_plot_ind = copy.deepcopy(fit)

        plt.plot(axis, y[plot_ind])
        plt.plot(axis, fit)
        plt.plot(points, averages, 'o', color='red')
        plt.show()

    # Calculations for the rest of spectra
    baseline = []
    for j in range(len(y)):
        if plot and j == plot_ind:
            baseline.append(baseline_plot_ind)
        else:
            averages = []
            for i in range(len(pos)):
                averages.append(np.mean(y[j][pos[i] - avg[i]:
                                             pos[i] + avg[i] + 1]))

            z = np.polyfit(points, averages, deg)  # polinomial fit
            f = np.poly1d(z)  # 1d polinomial
            fit = f(axis)

            if remove:
                baseline.append(y[j] - fit)
            else:
                baseline.append(fit)

    if dims == 1:
        baseline = baseline[0]

    return baseline


def valtoind(vals, x):
    """
    To translate the value in an axis to its index in the axis, basically
    searches for the position of the value. It approximates to the closest.

    :type vals: list[float]
    :param vals: List of values to be searched and translated.

    :type x: list[float]
    :param x: Axis.

    :returns: Index, or position, in the axis of the values in vals
    :rtype: list[int], int
    """
    vals = copy.deepcopy(vals)
    x_np = np.array(x)
    vals_np = np.array(vals)
    shape_dim = vals_np.ndim

    if shape_dim > 1:
        pos = [[np.argmin(np.abs(x_np - val)) for val in val_row] for val_row in vals_np]
    elif shape_dim == 1:
        pos = [np.argmin(np.abs(x_np - val)) for val in vals_np]
        vals[:] = [x_np[p] for p in pos]
    elif shape_dim == 0:
        pos = np.argmin(np.abs(x_np - vals_np))
        vals = x_np[pos]

    return pos


def areacalculator(y, x=None, limits=None, norm=False):
    """
    Area calculator using the data (x_data) and the limits in position, not
    values.

    :type y: list[float]
    :param y: Data to calculate area from.

    :type x: list[float]
    :param x: X axis of the data.

    :type limits: list[int]
    :param limits: Limits that define the areas to be calculated. Axis value.

    :type norm: bool
    :param norm: If True, normalized the area to the sum under all the curve.

    :returns: A list of areas according to the requested limits.
    :rtype: list[float]
    """
    dims = len(np.array(y).shape)

    if dims == 1:
        y = [y]

    if x is None:
        x = [i for i in range(len(y[0]))]

    limits = valtoind(limits, x)

    if len(np.array(limits).shape) == 1:
        limits = [limits]

    areas = np.zeros((len(y), len(limits)))
    y = np.array(y)
    for j, (start, end) in enumerate(limits):
        areas[:, j] = y[:, start:end].sum(axis=1)

    # Normalize if necessary
    if norm:
        areas /= y.sum(axis=1)[:, np.newaxis]

    if dims == 1:
        areas = areas[0]

    return areas.tolist()


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


def normsum(y, x=None, lims=None):
    """
    Normalizes the sum under the curve to 1, for single or multiple spectras.

    :type y: list[float]
    :param y: Single spectra or a list of them to normalize.

    :type lims: list[float, float]
    :param lims: Limits of the vector to normalize the sum. Default is `None`.
        For example, if ´lims = [N, M]´, the the sum under the curve betwween
        N and M is normalized to 1.

    :returns: Normalized data
    :rtype: list[float]
    """
    y = np.array(y, copy=True)
    dims = y.ndim

    if dims == 1:
        y = y.reshape(1, -1)

    if x is None:
        x = np.arange(y.shape[1])

    if lims is None:
        pos = [0, y.shape[1]]
    else:
        pos = valtoind(lims, x)

    s = y[:, pos[0]:pos[1]].sum(axis=1)
    y = y / s[:, np.newaxis]

    if dims == 1:
        y = y.ravel()

    return y


def normtoratio(y, r1, r2, x=None):
    """
    Normalizes a peak to the ratio value respect to another. That is, the peak
    found in the range of r1 is normalized to the ratio r1/(r1+r2).

    :type y: list[float]
    :param y: Single spectra or a list of them.

    :type r1: list[float, float]
    :param r1: Range of the first area according to the axis.

    :type r2: list[float, float]
    :param r2: Range of the second area according to the axis.

    :type x: list[float]
    :param x: Axis of the data. If `None` then it goes from 0 to N, where
        N is the length of the spectras.

    :returns: Normalized data
    :rtype: list[float]
    """
    y = np.array(y, copy=True)
    dims = y.ndim

    if dims == 1:
        y = y.reshape(1, -1)

    if x is None:
        r1, r2 = r1, r2
    else:
        r1 = valtoind(r1, x)
        r2 = valtoind(r2, x)

    a1 = y[:, r1[0]:r1[1]].sum(axis=1)
    a2 = y[:, r2[0]:r2[1]].sum(axis=1)
    total_areas = a1 + a2
    # To avoid nan results in the 0/0, it divides using np.where
    ratio = np.where(total_areas != 0, a1 / total_areas, 0)
    y_max = y.max(axis=1)[:, np.newaxis]
    # To avoid nan results in the 0/0, it divides using np.where
    m = np.where(y_max != 0, ratio[:, np.newaxis] / y_max, 0)
    y = y * m

    if dims == 1:
        y = y.ravel()

    return y.tolist()


def normtoglobalmax(y, globalmin=False):
    """
    Normalizes a list of spectras to the global max.

    :type y: list[float]
    :param y: List of spectras.

    :type globalmin: Bool
    :param globalmin: If `True`, the global minimum is reescaled to 0. Default
        is `False`.

    :returns: Normalized data
    :rtype: list[float]
    """
    y = np.array(y, copy=True)
    dims = y.ndim

    if dims > 1:
        # y_min, y_max = y.min(), y.max()
        y_min, y_max = np.min(y), np.max(y)
        if globalmin:
            y = y - y_min
            y_max -= y_min
        # y /= y_max
        y = y/y_max
    else:
        y_min, y_max = y.min(), y.max()
        if globalmin:
            y = y - y_min
        y = y / y_max
    return y.tolist()


def normtoglobalsum(y):
    """
    Normalizes a list of spectras to the global max sum under the curve. In
        other words, looks to the largest sum under the curve and sets it to 1,
        then the other areas are scaled in relation to that one.

    :type y: list[float]
    :param y: List of spectras.

    :returns: Normalized data
    :rtype: list[float]
    """
    y = np.array(y, copy=True)
    dims = y.ndim

    if dims > 1:
        sums = y.sum(axis=1)
        maxsum = sums.max()
        y = y / maxsum
    else:
        y = y / y.sum()
    return y


def interpolation(y, x, step=1, start=0, finish=0):
    """
    Interpolates data to a new axis.

    :type y: list[float]
    :param y: List of data to interpolate

    :type x: list[list[float]]
    :param x: list of the axises of the data

    :type step: float
    :param step: new step for the new axis

    :returns: Interpolated data and the new axis
    :rtype: list[float], list[float]
    """
    temp_y = copy.deepcopy(y)
    dims = len(np.array(y).shape)

    if start == 0 and start == finish:
        new_start = math.ceil(min(x))
        new_end = math.floor(max(x))
    else:
        new_start = start
        new_end = finish

    if min(x) > new_start:
        new_start = math.ceil(min(x))
    if max(x) < new_end:
        new_end = math.floor(max(x))

    x_new = np.arange(new_start, new_end + step, step)

    master_y = []

    if dims > 1:
        for i in range(len(temp_y)):
            this = interpolate.interp1d(x, temp_y[i])
            master_y.append(this(x_new))
    else:
        this = interpolate.interp1d(x, temp_y)
        master_y = this(x_new)

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
    :param all_targets: List of all real targets (making sure all groups are
        here).

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
        if g_count[i] == 0:
            print('No data with label (class) ' + str(i) + ' where found when calculating group scores. Check if the count is out of bounds or none samples of the class where included in the sample.\n')
        else:
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

    :returns: Score by comparing CM distances. Prediction using CM distances.
        X-axis coords of ths CMs. Y-axis coords of the Cms.
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
        if c[i] == 0:
            print('No samples for group ' + str(i) + ' were found when calculating the center of mass. Check if limits are out of range or samples of that target are missing.')
        else:
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

    :returns: Score by comparing MD distances. Prediction using MD distances.
        X-axis coords of ths CMs. Y-axis coords of the Cms.
    :rtype: list[float, list[int],list[float],list[float]]
    """
    x_p = list(x_p)
    y_p = list(y_p)
    g_n = int(max(tar) + 1)
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


def normtopeak(y, x, peak, shift=10):
    """
    Normalizes the spectras to a particular peak.

    :type y: list[float]
    :param y: Data to be normalized.

    :type x: list[float]
    :param x: x axis of the data

    :type peak: float
    :param peak: Peak position in x-axis values.

    :type shift: int
    :param shift: Range to look for the real peak. The default is 10.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    shift = int(shift)
    pos = valtoind(peak, x)

    if dims == 1:
        y = [y]

    for j in range(len(y)):
        section = y[j][pos - shift:pos + shift]
        y[j] = y[j] / max(section)

    if dims == 1:
        y = y[0]

    return y


def peakfinder(y, x=None, ranges=None, look=10):
    """
    Finds the location of the peaks in a single vector.

    :type y: list[float]
    :param y: Data to find a peak in. Single spectra.

    :type x: list[float]
    :param x: X axis of the data. If no axis is passed then the axis goes
        from 0 to N, where N is the length of the spectras. Default is `None`.

    :type ranges: list[[float, float]]
    :param ranges: Aproximate ranges of known peaks, if any. If no ranges are
        known or defined, it will return all the peaks that comply with the
        `look` criteria. If ranges are defined, it wont use the `look` criteria,
        but just for the absolute maximum within the range. Default is `None`.

    :type look: int
    :param look: Amount of position to each side to decide if it is a local
        maximum. The default is 10.

    :returns: A list of the index of the peaks found.
    :rtype: list[int]
    """
    peaks = []

    if x is None:
        x = [i for i in range(len(y))]

    if ranges is None:
        ranges = [0, len(y)]
    else:
        ranges = valtoind(ranges, x)

    if len(np.array(ranges).shape) == 1:
        ranges = [ranges]

    for i in ranges:
        section = y[i[0]:i[1]]

        m = np.max(section)
        for j in range(i[0], i[1]):
            if y[j] == m:
                peaks.append(int(j))


    if len(peaks) == 1:
        peaks = int(peaks[0])

    return peaks


def confusionmatrix(tt, tp, gn=None, plot=False, save=False, title='', ndw=True, cmm='Blues',
                    fontsize=20, figsize=(12, 15), ylabel='True', xlabel='Prediction',
                    filename='cm.png', rotation=(45, 0)):
    """
    Calculates and/or plots the confusion matrix for machine learning algorithm
    results.

    :type tt: list[float]
    :param tt: Real targets.

    :type tp: list[float]
    :param tp: Predicted targets.

    :type plot: boolean
    :param plot: If true, plots the matrix. The default is False

    :type gn: list[str]
    :param gn: Names, or lables , of the classification groups. Default is `None`.

    :type ndw: bool
    :param ndw: No data warning. If `True`, it warns about no havingdata to
        evaluate. Default is True.

    :type title: str
    :param title: Name of the matrix. Default is empty.

    :type cmm: str
    :param cmm: Nam eof the colormap (matplotlib) to use for the plot. Default
        is Blue.

    :type fontsize: int
    :param fontsize: Font size for the labels. The default is 20.

    :type figsize: Tuple
    :param figsize: Size of the image. The default is (12, 15).

    :type ylabel: str
    :param ylabel: Label for y axis. Default is `True`.

    :type xlabel: str
    :param xlabel: Label for x axis. Default is `Prediction`.

    :returns: The confusion matrix
    :rtype: list[float]
    """
    tt = np.array([int(i) for i in tt])
    tp = np.array([int(i) for i in tp])
    plot = bool(plot)

    if gn is None:
        gn = np.arange(int(np.max([tt, tp])) + 1)

    group_names = list(gn)
    gn = len(group_names)
    title = str(title)
    cmm = str(cmm)
    fontsize = int(fontsize)

    m = np.zeros((gn, gn))
    p = np.bincount(tt, minlength=gn)

    for i in range(gn):
        if p[i] == 0:
            if ndw is True:
                print(f'No data with label (class) {i} was found when making the confusion matrix. '
                      'Check if the count is out of bounds or none samples of the class were included in the sample.\n')
        else:
            m[i] = np.bincount(tp[tt == i], minlength=gn) / p[i]

    if plot or save:
        fig = plt.figure(tight_layout=True, figsize=figsize)
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
        plt.setp(ax.get_xticklabels(), rotation=rotation[0], ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=rotation[1], ha="right", rotation_mode="anchor")

        for i in range(gn):
            for j in range(gn):
                ax.text(j, i, "{:.2f}".format(round(m[i][j], 2)), ha='center', va='center', color='black')

        if plot:
            plt.show()

        if save:
            plt.savefig(filename)

    return m


def avg(y):
    """
    Calculates the average vector from a list of vectors.

    :type y: list[float]
    :param y: List of vectors.

    :returns: The average of the vectors in the list.
    :rtype: list[float]
    """
    dims = len(np.array(y).shape)  # detect dimensions

    if dims > 1:
        avg_data = np.mean(y, axis=0)
    else:
        avg_data = sum(y)/len(y)

    return avg_data


def sdev(y):
    """
    Calculates the standard deviation for each bin from a list of vectors.

    :type y: list[float]
    :param y: List of vectors.

    :returns: Standard deviation curve
    :rtype: list[float]
    """
    y = np.array(y)
    if y.ndim > 1:
        curve_std = np.std(y, axis=0)
    else:
        curve_std = sta.stdev(y)
    return curve_std


def median(y):
    """
    Calculates the median vector of a list of vectors.

    :type y: list[float]
    :param y: List of vectors.

    :returns: median curve
    :rtype: list[float]
    """
    dims = len(np.array(y).shape)
    median = []
    if dims > 1:
        median = np.median(y, axis=0)
    else:
        median = np.median(y)
    return median


def lorentzfit(y=[0], x=[0], pos=0, look=5, shift=2, gamma=5, alpha=1, manual=False):
    """
    Fits peak as an optimization problem or manual fit for Lorentz distirbution,
    also known as Cauchy. A curve `y` is only mandatory if the optimixzation is
    needed (manual=False, default). If no axis 'x' is defined, then a default
    axis is generated for both options.

    :type y: list[float]
    :param y: Data to fit. Single vector.

    :type x: list[float]
    :param x: x axis.

    :type pos: int
    :param pos: X axis position of the peak.

    :type look: int
    :param look: axis positions to look to each side in axis units. The default is 5.

    :type shift: int
    :param shift: Possible axis shift of the peak in axis units. The default is 2.

    :type gamma: float
    :param gamma: Lorentz fit parameter. The default is 5.

    :type alpha: float
    :param alpha: Multiplier of the fitting. The maximum value fo the fitting
        is proportional to this value, but is not necesarly its value. The
        default is 1.

    :type manual: boolean
    :param manual: If `True`, 1 curve will be generated using the declared
        parameter `gamma` and perform a manual fit. Default is `False`.

    :returns: Fitted curve.
    :rtype: list[float]
    """
    ax = list(x)
    y = list(y)

    if manual:
        if ax == [0]: # if no axis is passed
            ax = range(-100, 100)

        fit = [alpha/(np.pi*gamma*(1+((i-pos)/gamma)**2)) for i in ax]

    else:
        if ax == [0]: # if no axis is passed
            # ax = [i for i in range(len(y))]
            ax = range(len(y))

        s = int(shift/abs(ax[1]-ax[0]))
        look = int(look/abs(ax[1]-ax[0]))

        pos = int(valtoind(pos, ax))

        for k in range(pos-s, pos+s):
                if y[k] > y[pos]:
                    pos = k
        p = ax[pos]

        def objective(x):
            fit, error = 0, 0
            ppos = valtoind(x[2], ax)

            for i in range(ppos-look, ppos+look+1):  # for all the points
                fit = x[0]*(1/(np.pi*x[1]*(1+((ax[i]-x[2])/x[1])**2)))
                error += (fit-y[i])**2

            return error**0.5

        def constraint1(x):
            return 0

        x0 = np.array([alpha, gamma, p])  # master vector to optimize, initial values
        bnds = [[0.00000001, max(y)*9999999], [0.1, 9999], [p-s, p+s]]
        con1 = {'type': 'ineq', 'fun': constraint1}
        cons = ([con1])
        solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x

        fit = [x[0]*(1/(np.pi*x[1]*(1+((ax[l]-x[2])/x[1])**2))) for l in range(len(ax))]

    return fit


def gaussfit(y=[0], x=[0], pos=0, look=5, shift=2, sigma=4.4, alpha=1, manual=False, params=False):
    """
    Fits peak as an optimization problem or manual fit. A curve `y` is only
    mandatory if the optimixzation is needed (manual=False, default). If no
    axis 'ax' is defined, then a default axis is generated for both options.

    :type y: list[float]
    :param y: Data to fit. Single vector.

    :type x: list[float]
    :param x: x axis.

    :type pos: int
    :param pos: Peak index to fit to.

    :type look: int
    :param look: axis positions to look to each side in axis units. The default is 5.

    :type shift: int
    :param shift: Possible axis shift of the peak in axis units. The default is 2.

    :type sigma: float
    :param sigma: Sigma value for Gaussian fit. The default is 4.4.

    :type alpha: float
    :param alpha: Multiplier of the fitting and initial value of the optimizer
        when `manual=False`. The maximum value fo the fitting is proportional
        to this value, but is not necesarly its value. The default is 1.

    :type manual: boolean
    :param manual: If `True`, 1 curve will be generated using the declared
        parameter `sigma` and perform a manual fit. Default is `False`.

    :type params: boolean
    :param params: If `True`, it returns the parameters of the fit in the
        order of `alpha`, `sigma` and `peak position`. Default is `False`.

    :returns: Fitted curve.
    :rtype: list[float]
    """
    ax = list(x) # from now foreward, x is the variable for the optimization.

    if manual:
        if ax == [0]: # if no axis is passed
            ax = [i for i in range(-100, 100)]

        fit = []
        for i in ax:
            fit.append((alpha/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((i-pos)/sigma)**2))))

    else:
        if ax == [0]: # if no axis is passed
            ax = [i for i in range(len(y))]

        y = list(y)

        pos = int(valtoind(pos, ax))

        for i in range(len(y)):
            if y[i] < 0:
                y[i] = 1

        s = int(shift/abs(ax[1]-ax[0]))
        look = int(look/abs(ax[1]-ax[0]))

        for k in range(pos - s, pos + s):
            if y[k] > y[pos]:
                pos = k

        p = ax[pos]

        def objective(x):
            fit = 0
            error = 0
            ppos = valtoind(x[2], ax)

            for i in range(ppos - look, ppos + look + 1):  # for all the points
                fit = (x[0]/(x[1]*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((ax[i]-x[2])/x[1])**2)))
                error += (fit - y[i])**2  # total error

            return error**0.5

        def constraint1(x):
            return 0

        x0 = np.array([alpha, sigma, p], dtype='object')  # master vector to optimize, initial values
        bnds = [[0.0000001, max(y)*99999], [0.0000001, 1000], [(p-s), (p+s)]]
        con1 = {'type': 'ineq', 'fun': constraint1}
        cons = ([con1])
        solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x

        fit = []
        for l in range(len(ax)):  # for all the points
            fit.append(((x[0]/(x[1]*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((ax[l]-x[2])/x[1])**2)))))

    if params:
        return x
    else:
        return fit


def studentfit(y=[0], x=[0], pos=0, v=0.01, alpha=1, look=5, shift=2, manual=False):
    """
    Fits peak as an optimization problem.

    :type y: list[float]
    :param y: Data to fit. Single vector.

    :type x: list[float]
    :param x: x axis.

    :type pos: int
    :param pos: Peak position to fit, in axis values.

    :type v: float
    :param v: Student fit parameter. The default is 0.01.

    :type alpha: float
    :param alpha: Multiplier of the fitting. The maximum value fo the fitting
        is proportional to this value, but is not necesarly its value. The
        default is 1.

    :type look: int
    :param look: axis positions to look to each side in axis units. The default is 5.

    :type shift: int
    :param shift: Possible axis shift of the peak in axis units. The default is 2.

    :type manual: boolean
    :param manual: If `True`, 1 curve will be generated using the declared
        parameter `sigma` and perform a manual fit. Default is `False`.

    :returns: Fitted curve.
    :rtype: list[float]
    """

    # initial guesses
    ax = list(x)
    y = list(y)
    v = float(v)

    if manual:
        if ax == [0]: # if no axis is passed
            ax = [i/10 for i in range(-100, 100)]
        fit = []
        for i in ax:  # for all the points
            a = gamma((v+1)/2)
            b = np.sqrt(np.pi*v)*gamma(v/2)
            c = 1+((i-pos)**2)/v
            d = -(v+1)/2
            fit.append(alpha*(a/b)*(c**d))

    else:
        pos = int(valtoind(pos, ax))
        ax = [i for i in range(len(y))] # it only works with this axis, idkw

        s = int(shift/abs(ax[1]-ax[0]))
        look = int(look/abs(ax[1]-ax[0]))

        for k in range(pos - s, pos + s):
                if y[k] > y[pos]:
                    pos = k

        def objective(x):
            fit = 0
            error = 0
            for i in range(pos - look, pos + look):  # for all the points
                a = gamma((x[1]+1)/2)
                b = np.sqrt(np.pi*x[1])*gamma(x[1]/2)
                c = 1+((ax[i]-x[2])**2)/x[1]
                d = -(x[1]+1)/2
                fit = x[0]*(a/b)*(c**d)

                error += (fit-y[i])**2
            return error**0.5

        def constraint1(x):
            # return x[0]*x[1]*x[2]
            return 0


        x0 = np.array([alpha, v, pos])  # master vector to optimize, initial values

        bnds = [[0, max(y)*99999], [0.0001, 99], [pos-s, pos+s]]
        con1 = {'type': 'ineq', 'fun': constraint1}
        cons = ([con1])
        solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x

        fit = []

        for i in ax:  # for all the points
            a = gamma((x[1]+1)/2)
            b = np.sqrt(np.pi*x[1])*gamma(x[1]/2)
            c = 1+((i-x[2])**2)/x[1]
            d = -(x[1]+1)/2

            fit.append(x[0]*(a/b)*(c**d))

    return fit


def decbound(x_points, y_points, groups, limits=None, divs=0.5):
    """
    Calculates the Decision Boundaries.

    :type x_points: list[float]
    :param x_points: X coordinates of each point.

    :type y_points: list[float]
    :param y_points: Y coordinates of each point.

    :type groups: list[int]
    :param groups: List of targets for each point.

    :type limits: list[float]
    :param limits: Plot and calculation limits. The default is 'None'.

    :type divs: float
    :param divs: Resolution. The default is 0.01.

    :returns: The decision boundaries.
    :rtype: list[float]
    """
    css, cmp, cmx, cmy = cmscore(x_points, y_points, groups)

    divs = float(divs)  # step

    if limits is None :
        map_x = [int((min(x_points)-1)/divs), int((max(x_points)+1)/divs)]
        map_y = [int((min(y_points)-1)/divs), int((max(y_points)+1)/divs)]
    else:
        map_x = list(np.array(limits[:2]) / divs)  # mapping limits
        map_y = list(np.array(limits[2:]) / divs)

    x_divs = int(map_x[1] - map_x[0])  # mapping combining 'divs' & 'lims'
    y_divs = int(map_y[1] - map_y[0])

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
        A.append(v)

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


def decdensity(x, y, groups, limits=None, divs=0.5, th=2):
    """
    Calculates the density decision map from a cluster mapping.

    :type x: list[float]
    :param x: X coordinates of each point.

    :type y: list[float]
    :param y: Y coordinates of each point.

    :type groups: list[int]
    :param groups: List of targets for each point.

    :type limits: list[float]
    :param limits: Plot and calculation limits. The default is 'None'.

    :type divs: float
    :param divs: Resolution ti calculate density. The default is 0.5.

    :type th: int
    :param th: Threshold from where a area is defined as a certain group.

    :returns: The density decision map.
    :rtype: list[float]
    """

    divs = float(divs)

    if limits is None :
        map_x = [int((min(x)-1)/divs), int((max(x)+1)/divs)]
        map_y = [int((min(y)-1)/divs), int((max(y)+1)/divs)]
    else:
        map_x = list(np.array(limits[:2]) / divs)  # mapping limits
        map_y = list(np.array(limits[2:]) / divs)

    x_p = list(x)
    y_p = list(y)

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

                if count > th:
                    pmap[j][i] = count
                else:
                    pmap[j][i] = 0

        maximum = max(np.array(pmap).flatten())

        if maximum == 0:
            print('No density was found for group ' + str(l) + '. Check if the range is out of bounds or targets with this value are missing.\n')
            pmap = np.array(pmap)
        else:
            pmap = np.array(pmap) / maximum

        master.append(pmap)
    return master


def isaxis(y):
    """
    Detects if there is an axis in the data.

    :type data: list[float]
    :param data: Data containing spectras an possible axis.

    :returns: True if there is axis.
    :rtype: bool
    """
    features = list(y)
    axis = features[0]
    if len(np.array(y).shape) == 1:
        axis = features
    return all(axis[i] <= axis[i + 1] for i in range(len(axis) - 1))



def trim(y, start=0, finish=0):
    """
    Deletes columns in a list from start to finish.

    :type y: list
    :param y: Data to be trimmed.

    :type start: int, optional
    :param start: Poistion of the starting point. The default is 0.

    :type finish: int, optional
    :param finish: Position of the ending point (not included). The default is 0.

    :returns: Trimmed data.
    :rtype: list[]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)

    if dims == 1:
        y = [y]

    final = []
    if finish == 0 or finish > len(y[0]):
        finish = len(y[0])
    t = finish - start

    for j in y:
        temp = j
        for _ in range(t):
            temp = np.delete(temp, start, 0)
        final.append(list(temp))
    y = final

    if dims == 1:
        y = y[0]

    return y


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


def shiftref(ref_data, ref_axis, ref_peak=520, mode=1, plot=True):
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

    :type plot: bool
    :param plot: If True plots a visual aid. The default is True.

    :returns: Shift amount
    :rtype: float
    """
    ref_data = list(ref_data)
    ref_axis = list(ref_axis)
    # ref_peak = valtoind(ref_peak, ref_axis)
    mode = int(mode)
    plot = bool(plot)

    fit = []  # fit curves(s), if selected
    shift = []  # axis shift array

    dims = len(np.array(ref_data).shape)
    if dims > 1:
        for i in range(len(ref_data)):  # depending on the mode chosen...
            if mode == 1:
                fit.append(lorentzfit(y=ref_data[i], x=ref_axis[i], pos=ref_peak))
            if mode == 2:
                fit.append(gaussfit(y=ref_data[i], x=ref_axis[i], pos=ref_peak))
            if mode == 0:
                fit.append(ref_data[i])
    else:
        if mode == 1:
            fit.append(lorentzfit(y=ref_data, x=ref_axis, pos=ref_peak))
        if mode == 2:
            fit.append(gaussfit(y=ref_data, x=ref_axis, pos=ref_peak))
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
        if dims > 1:
            plt.figure()  # figsize = (16,9)
            for i in range(len(ref_data)):
                plt.plot(ref_axis[i], ref_data[i], linewidth=2, label='Original' + str(i))
                plt.plot(ref_axis[i], fit[i], linewidth=2, label='Fit' + str(i), linestyle='--')
            plt.axvline(x=ref_peak, ymin=0, ymax=max(ref_data[0]), linewidth=2, color="red", label=ref_peak)
            plt.axvline(x=ref_peak - peakshift, ymin=0, ymax=max(ref_data[0]), linewidth=2, color="yellow",
                        label="Meas. Max.")
        else:
            plt.plot(ref_axis, ref_data, linewidth=2, label='Original' + str(i))
            plt.plot(ref_axis, fit[0], linewidth=2, label='Fit', linestyle='--')
            plt.axvline(x=ref_peak, ymin=0, ymax=max(ref_data), linewidth=2, color="red", label=ref_peak)
            plt.axvline(x=ref_peak - peakshift, ymin=0, ymax=max(ref_data), linewidth=2, color="yellow",
                        label="Meas. Max.")
        plt.xlim(ref_peak - 15, ref_peak + 15)
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

        group_names.append(str(var)+' $<$ ' + str(group_limits[0]))
        for i in range(0, len(group_limits) - 1):
            group_names.append(str(group_limits[i]) + ' $<=$ '+str(var)+' $<$ ' + str(group_limits[i + 1]))
        group_names.append(str(max(group_limits)) + ' $<=$ '+str(var))

    return class_targets, group_names


def subtractref(data, ref, axis=0, alpha=0.9, sample=0, lims=[0, 0], plot=False):
    """
    Subtracts a reference spectra from the measurements.

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

    :type lims: list[int]
    :param lims: Limits of the plot.

    :type plot: bool
    :param plot: To plot or not a visual aid. The default is True.

    :returns: Data with the subtracted reference.
    :rtype: list[float]
    """
    data = copy.deepcopy(data)
    dims = len(np.shape(data))
    data_sub = []

    if dims == 1:
        data = [data]

    for i in data:
        data_sub.append(np.array(i) - np.array(ref) * alpha)

    if plot:
        if axis == 0:
                axis = [i for i in range(len(ref))]
        else:
            axis = list(axis)

        plt.plot(axis, data_sub[sample], linewidth=1, label='Corrected')
        plt.plot(axis, np.array(ref)*alpha, linewidth=1, label='Air*Alpha', linestyle='--')
        plt.plot(axis, data[sample], linewidth=1, label='Original', linestyle='--')
        if lims[0] < lims[1]:
            plt.gca().set_xlim(lims[0], lims[1])
        plt.legend(loc=0)
        plt.ylabel('a.u.')
        plt.xlabel('Shift (cm-1)')
        plt.show()

    if dims == 1:
        data_sub = data_sub[0]

    return data_sub


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

    :returns: Pearson plot in a 2d list.
    :rtype: list[float]
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
        plt.xticks(rotation='vertical')
        plt.show()

    return pears

def spearman(data, labels=[], cm="seismic", fons=20, figs=(20, 17),
             tfs=25, ti="Spearman", plot=True):
    """
    Calculates Spearman matrix and plots it.

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

    :returns: Spearman plot in a 2d list.
    :rtype: list[float]
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
        plt.xticks(rotation='vertical')
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
    :param marks: Marker size. The default is 100.

    :returns: Grau plot in a 2d list.
    :rtype: list[float]
    """
    labels = list(labels)
    cm = str(cm)
    fontsize = float(fons)  # plot font size
    figsize = list(figs)  # plot size
    titlefs = float(tfs)  # title font size
    title = str(ti)  # plot name
    marker = str(marker)  # market style
    marks = float(marks)  # marker size

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
                        c=mse, cmap=cm, s=marks, marker=marker)
        plt.colorbar(sc)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xlim(0, max(x_ticks))
        ax.get_xticklabels()[0].set_fontweight('bold')
        plt.xticks(rotation=90)
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


def moveavg(y, move=2):
    """
    Calculate the moving average of a single or multiple vectors.

    :type y: list[float]
    :param y: Data to calculate the moving average. Single or multiple vectors.

    :type move: int
    :param move: Average range to each side (total average = move + 1).

    :returns: Smoothed vector(s).
    :rtype: list[float]
    """
    move = int(move)
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    avg = []

    if dims == 1:
        y = [y]
    data_len = len(y[0])

    for j in range(len(y)):
        temp = []
        for i in range(move, data_len - move):
            temp.append(np.mean(y[j][i - move: i + move + 1]))
        for i in range(move):
            temp.append(temp[-1])
            temp.insert(0, temp[0])
        avg.append(temp)

    if dims == 1:
        avg = avg[0]

    return avg


def plot2dml(train, test=[], names=['D1', 'D2', 'T'], train_pred=[],
             test_pred=[], labels=[], title='', xax='x', yax='y', fs=15,
             lfs=10, loc='best', size=20, xlim=[], ylim=[], plot=True, save=False,
             filename='lda.png'):
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

    :type plot: boolean
    :param plot: If True it plot. Only for test purposes.

    :type save: boolean
    :param save: If true, it saves the plot in the filename directory.
    The default is False.

    :type filename: str
    :param filename: Directory provided to save the plot.

    :returns: Plot
    """
    # marker = ['o', 'v', 's', 'd', '*', '^', 'x', '+', '.',
    #           'o', 'v', 's', 'd', '*', '^', 'x', '+', '.']
    marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
              'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    marker_test = ["D", "D", "D", "D", "D", "D", "D", "D", "D",
                   "D", "D", "D", "D", "D", "D", "D", "D", "D"]
    color = ['red', 'green', 'blue', 'grey', 'orange', 'olive', 'lime',
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
            plt.scatter(test[names[0]][i], test[names[1]][i], alpha=0.7,
                        s=size-1, linewidths=0.75, color=color[group],
                        marker=marker_test[group], edgecolor=ec)

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
        plt.tight_layout()
        if save:
            plt.savefig(filename)
        plt.show()
    return plot


def stackplot(y, offset, order=None, xlabel='', ylabel='', title='', cmap='Spectral',
              figsize=(6, 9), fs=20, lw=1, xlimits=None, plot=True):
    """
    Plots a stack plot of selected spectras.

    :type y: list[float]
    :param y: Data to plot. Must be more than 1 vector.

    :type offset: float
    :param offset: displacement, or difference, between each curve.

    :type order: list[int]
    :param order: Order of the curves in which they are plotted. If `None`,
        is the order as they appear in the list.

    :type xlabel: str
    :param xlabel: Label of axis.

    :type ylabel: str
    :param ylabel: Label of axis.

    :type title: str
    :param title: Title of the plot.

    :type cmap: str
    :param cmap: Colormap, according to matplotlib options.

    :type figsize: tuple
    :param figsize: Size of the plot. Default is (3, 4.5)

    :type fs: float
    :param fs: Font size. Default is 20.

    :type lw: float
    :param lw: Linewidth of the curves.

    :type xlim: list[float]
    :param xlim: Plot limits for x-axis. If `None` it plots all.

    :type plot: bool
    :param plot: If True it plot. Only for test purposes.

    :returns: plot
    :rtype: bool
    """

    base = [offset for _ in range(len(y[0]))]

    cmap = plt.cm.get_cmap(cmap)
    color = []
    for i in range(len(y)):
        color.append(cmap(i/(len(y)-1)))

    if plot:
        plt.figure(figsize=figsize)
        for i in range(len(y)):
            plt.plot(np.array(y[i]) + np.array(base)*i, color=color[i], lw=lw)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.xlabel(xlabel, fontsize=fs)
        plt.ylabel(ylabel, fontsize=fs)
        plt.title(title, fontsize=fs)
        if xlimits:
            plt.xlim(xlimits[0], xlimits[1])
        plt.show()

    return plot


def cosmicmp(y, alpha=1, avg=2):
    """
    It identifies CRs by comparing similar spectras and paring in matching
    pairs. Uses randomnes of CRs (S. J. Barton, B. M. Hennelly, https://doi.org/10.1177/0003702819839098)

    :type y: list[float]
    :param y: List of spectras to remove cosmic rays.

    :type alpha: float
    :param alpha: Factor to modify the criteria to identify a cosmic ray.

    :type avg: int
    :param avg: Moving average window.

    :returns: Data with removed cosmic rays.
    :rtype: list[float]
    """
    data = copy.deepcopy(y)

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


def cosmicdd(y, th=100, asy=0.6745, m=5):
    """
    It identifies CRs by detrended differences, the differences between a
    value and the next (D. A. Whitaker and K. Hayes, https://doi.org/10.1016/j.chemolab.2018.06.009).

    :type y: list[float]
    :param y: List of spectras to remove cosmic rays.

    :type th: float
    :param th: Factor to modify the criteria to identify a cosmic ray.

    :type asy: float
    :param asy: Asymptotic bias correction

    :type m: float
    :param m: Number of neighbor values to use for average.

    :returns: Data with removed cosmic rays.
    :rtype: list[float]
    """
    data = copy.deepcopy(y)

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


def cosmicmed(y, sigma=1.5):
    """
    Precise cosmic ray elimination for measurements of the same point or very
    similar spectras.

    :type y: list[float]
    :param y: List of spectras to remove cosmic rays.

    :type sigma: float
    :param sigma: Factor to modify the criteria to identify a cosmic ray.
        Multiplies the median of each bin.

    :returns: Data with removed cosmic rays.
    :rtype: list[float]
    """
    solved = copy.deepcopy(y)
    acq = len(solved)
    length = len(solved[0])

    med = median(solved)

    for i in range(acq):
        for j in range(length):
            if y[i][j] > sigma*med[j]:
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
    length = int(length)
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


def minmax(y):
    """
    Calculates the vectors that contain the minimum and maximum values of each
    bin from a list of vectors.

    :type y: list
    :param y: List of vectors to calculate the minimum and maximum vectors.

    :returns: minimum and maximum vectors.
    :rtype: list[float]
    """
    minimum = [min(col) for col in zip(*y)]
    maximum = [max(col) for col in zip(*y)]
    return minimum, maximum


def fwhm(y, x, peaks, alpha=0.5, s=10):
    """
    Calculates the Full Width Half Maximum of specific peak or list of
    peaks for a single or multiple spectras.

    :type y: list
    :param y: spectrocopic data to calculate the fwhm from. Single vector or
        list of vectors.

    :type x: list
    :param x: Axis of the data. If none, then the axis will be 0..N where N
        is the length of the spectra or spectras.

    :type peaks: float or list[float]
    :param peaks: Aproximate axis value of the position of the peak. If single
        peak then a float is needed. If many peaks are requierd then a list of
        them.

    :type alpha: float
    :param alpha: multiplier of maximum value to find width ratio. Default is 0.5
        which makes this a `full width half maximum`. If `alpha=0.25`, it would
        basically find the `full width quarter maximum`. `alpha` should be
        ´0 < alpha < 1´. Default is 0.5.

    :type s: int
    :param s: Shift to sides to check real peak. The default is 10.

    :type interpolate: boolean
    :param interpolate: If True, will interpolte according to `step` and `s`.

    :returns: A list, or single float value, of the fwhm.
    :rtype: float or list[float]
    """
    dims = len(np.array(y).shape)

    if dims == 1:
        y = [y]

    ind = valtoind(peaks, x)
    dims_peaks = len(np.array(peaks).shape)
    if dims_peaks < 1:
        ind = [ind]

    r_fwhm = []
    for h in y:

        fwhm = []
        for j in range(len(ind)):
            for i in range(ind[j] - s, ind[j] + s):
                if h[i] > h[ind[j]]:
                    ind[j] = i

            h_m = h[ind[j]]*alpha # half maximum

            temp = float('inf')
            left = 0
            for i in range(ind[j]):
                delta = abs(h[ind[j]-i] - h_m)
                if temp > delta:
                    temp = delta
                    left = ind[j]-i

            temp = float('inf')
            right = 0
            for i in range(len(x)-ind[j]):
                delta = abs(h[ind[j]+i] - h_m)
                if temp > delta:
                    temp = delta
                    right = ind[j]+i

            if dims_peaks < 1:
                fwhm = x[right] - x[left]
            else:
                fwhm.append(x[right] - x[left])

        if dims > 1:
            r_fwhm.append(fwhm)
        else:
            r_fwhm = fwhm

    return r_fwhm


def asymmetry(y, x, peak, s=5, limit=10):
    """
    Compares both sides of a peak, or list of peaks, and checks how similar
    they are. It does this by calculating the MRSE and indicating which side is
    larger or smaller by area. If it is a negative (-), then left side is
    smaller.

    :type y: list
    :param y: spectrocopic data to calculate the asymmetry from. Single vector or
        list of vectors.

    :type x: list
    :param x: Axis of the data. If none, then the axis will be 0..N where N is
        the length of the spectra or spectras.

    :type peak: float, list[float]
    :param peak: Aproximate axis value of the position of the peak. If sinlge
        peak then a float is needed. If many peaks are requierd then a list of
        them.

    :type s: int
    :param s: Shift to sides to check real peak. The default is 5.

    :type limit: int
    :param limit: Comparison limits to each side of the peak. Default is 10.

    :returns: R2 value comparing both sides of the peak and a sign to tell if
        either left side is smaller (negative) or tight side is smaller
        (positive, no sign included).
    :rtype: list[float]
    """
    dims = len(np.array(y).shape)
    index = valtoind(peak, x)

    if dims == 1:
        y = [y]

    final = []
    for h in y:
        for i in range(index - s, index + s):
            if h[i] > h[index]:
                index = i
        diff_nom = 0
        diff_abs = 0
        for i in range(limit):
            diff_nom += h[index - i] - h[index + i]
            diff_abs += (h[index - i] - h[index + i])**2
        if diff_nom < 0: # left side is smaller -> right larger
            diff_abs = np.sqrt(diff_abs/(2*limit))*(-1)
        final.append(diff_abs)

    if dims == 1:
        final = final[0]

    return final


def rwm(y, ws, plot=False):
    """
    Computes the median of an array in a running window defined by the user.
    The window size needs to be a 2D tuple or list, with the first element being the
    length of the spectra and the second the total width that will be taken into account
    for the statistics.

    :type y: numpy array
    :param y: The spectras

    :type ws: tuple/list
    :param ws: Window size parameters

    :type plot: boolean
    :param plot: If you want to plot the new spectra change to True

    :returns: Array containing the computed 1D spectra.
    :rtype: numpy array[float]
    """
    # Check dimensions
    if np.shape(ws)[0] != 2:
        raise ValueError("The window must be 2D")
    if (ws[1] % 2) == 0:
        raise ValueError("The width must be an odd number")

    # First pad the end and the beginning by duplicating the first and last columns from the spectras
    pad_length = int((ws[1]-1)/2)
    data_new = np.c_[y, y[:,-pad_length:]]
    data_new2 = np.c_[y[:,:pad_length], data_new]

    # Get the windows
    window = np.lib.stride_tricks.sliding_window_view(data_new2, ws)

    # Create new spectra from the median of each window
    new_spectra = []
    for i in window[0]:
        new_spectra.append(np.median(i))
    new_spectra = np.asarray(new_spectra)

    if plot:
        plt.plot(new_spectra)
        plt.show()

    return new_spectra


def typical(y):
    """
    Looks for the typical spectra of a dataset. In other words, it calculates
    the average spectra and looks for the one that is closer to it in relation
    to standard deviation. Do not confuse with a `Representative` spectra.

    :type y: list[list[float]]
    :param y: A list of spectras.

    :returns: The typical spectra of the set.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    m = avg(y)

    std = float('inf')
    typical = []
    for i in y:
        temp = sum(sdev([m, i]))
        if temp < std:
            std = temp
            typical = i

    return typical


def issinglevalue(y):
    """
    Checks if a vector, or a list of vectors, is composed of the same value
    (single value vector). Not to be confused with a single element vector
    were the length of the vector is 1.

    :type y: list
    :param y: Vector, or list of vectors, that needs to be checked.

    :returns: True if contains the same value. False if there are different. If
        `y` is a list of vectors then it returns a list of booleans with the
        respective answer.
    :rtype: bool
    """
    dims = len(np.shape(y))
    if 1 < dims: # checks if y is a list of lists
        isequal = [len(set(i))==1 for i in y] # applies the check to each list in y
    elif dims == 1:
        isequal = len(set(y))==1 # checks if all elements in the list are equal
    return isequal


def mahalanobis(v):
    """
    Calculates the Mahalanobis distance for a groups of vectors to the center
    of mass, or average coordinates.

    :param v: vectors to calculate the distance
    :type v: list

    :returns: List of the respectve distances.
    :rtype: list
    """
    mean = avg(v)
    cov = [[0 for _ in range(len(v[0]))] for _ in range(len(v[0]))]
    length = len(v)

    for i in range(len(cov)):
        for j in range(len(cov)):
            for k in range(len(v)):
                cov[i][j] += (v[k][i]-mean[i])*(v[k][j]-mean[j])
            cov[i][j] = cov[i][j]/length

    inverse = np.linalg.inv(cov)
    mahdist = []
    for i in v:
        mahdist.append(np.sqrt(np.dot(np.dot(np.transpose((i-mean)), inverse), (i-mean))))

    return mahdist


def representative(y):
    """
    Looks for the representative spectra of a dataset. In other words, it calculates
    the median spectra and looks for the one that is closer to it in relation
    to standard deviation. Do not confuse with a `Typical` spectra.

    :type y: list[list[float]]
    :param y: A list of spectras.

    :returns: The representative spectra of the set.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    m = median(y)

    std = float('inf')
    reptve = []
    for i in y:
        temp = sum(sdev([m, i]))
        if temp < std:
            std = temp
            reptve = i

    return reptve


def voigtfit(y=None, x=None, pos=0, look=5, shift=2, gamma=5, sigma=4, alpha=1, manual=False):
    """
    Fits peak as an optimization problem or manual fit for Voigt distirbution,
    also known as a convoluted Gaussian-Lorentz curve. A curve `y` is only
    needed for the optimization (manual=False, default). If no axis `x` is
    defined, then a default axis is generated for both options. It is reccomended to
    Normalize the data before fitting.

    :type y: list[float]
    :param y: Data to fit. Single vector.

    :type x: list[float]
    :param x: x axis.

    :type pos: int
    :param pos: X axis position of the peak.

    :type look: int
    :param look: axis positions to look to each side in axis units. The default is 5.

    :type shift: int
    :param shift: Possible axis shift of the peak in axis units. The default is 2.

    :type gamma: float
    :param gamma: Initial value of fit. The default is 5.

    :type sigma: float
    :param sigma: Initial value of fit. The default is 4.

    :type alpha: float
    :param alpha: Multiplier of the fitting. The maximum value fo the fitting
        is proportional to this value, but is not necesarly its value. The
        default is 1.

    :type manual: boolean
    :param manual: If `True`, 1 curve will be generated using the declared
        parameter `gamma` and perform a manual fit. Default is `False`.

    :returns: Fitted curve.
    :rtype: list[float]
    """
    ax = x

    if manual:
        if ax is None: # if no axis is passed
            ax = [i for i in range(-100, 100)]

        s = int(shift/abs(ax[1]-ax[0]))

        fit = []
        for i in ax:
            lor = 1/(np.pi*gamma*(1+((i-pos)/gamma)**2))
            gau = (1/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((i-pos)/sigma)**2)))
            fit.append(gau*lor*alpha)

    else:
        if ax is None: # if no axis is passed
            ax = [i for i in range(len(y))]

        s = int(shift/abs(ax[1]-ax[0]))
        look = int(look/abs(ax[1]-ax[0]))

        pos = int(valtoind(pos, ax))

        for k in range(pos-s, pos+s):
                if y[k] > y[pos]:
                    pos = k
        p = ax[pos]

        def objective(x):
            ppos = valtoind(x[3], ax)
            error = 0

            for i in range(ppos-look, ppos+look):  # for all the points
                gau = (1/(x[2]*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((ax[i]-x[3])/x[2])**2)))
                lor = (1/(np.pi*x[1]*(1+((ax[i]-x[3])/x[1])**2)))
                fit = gau*lor*x[0]

                error += (fit-y[i])**2  # total

            return error**0.5

        def constraint1(x):
            return 0

        x0 = np.array([alpha, gamma, sigma, p])  # master vector to optimize, initial values
        bnds = [[0.00000001, max(y)*99999], [0.01, 9999], [0.01, 9999], [(p-s), (p+s)]]
        con1 = {'type': 'ineq', 'fun': constraint1}
        cons = ([con1])
        solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x
        fit = []

        for l in range(len(ax)):  # for all the points
            lor = (1/(np.pi*x[1]*(1+((ax[l]-x[3])/x[1])**2)))
            gau = (1/(x[2]*(np.sqrt(2*np.pi))))*(np.exp(-0.5*(((ax[l]-x[3])/x[2])**2)))
            fit.append(gau*lor*x[0])

    return np.array(fit)


def autocorrelation(y, x, lag, lims=None, normalization=False, average=False):
    """
    Calculates the correlation of a signal with a delayed copy of itself as a
    function of the lag. In other words, it calculates the PEarsoon coefficient
    for a vector with itself shifted by `lag` positions.

    :type y: list[float]
    :param y: Data to fit. Single or list of vectors.

    :type x: list[float]
    :param x: x axis.

    :type lag: int
    :param lag: positions shifetd to be analyzed, not in x-axis values.

    :type lims: list[float, float]
    :param lims: limits of the data to autocorrelate. Defauilt is `None` and
        will analyze with all the vector (`[0:len(y)]`).

    :type normalization: boolean
    :param normalization: divides the autocorrelation value by the product of
        the standard deviation of the series in `t` and `t+lag`.

    :type average: boolean
    :param average: divides the autocorrelation value by the length of the
        signal.

    :returns: autocorrelation values, either in a list or single value.
    :rtype: float, list[float]
    """
    y = copy.deepcopy(y)

    dims = len(np.array(y).shape)
    if dims <= 1:
        y = [y]

    if lims is None:
        lims=[0, len(y[0])]
    lims = valtoind(lims, x)

    c = []
    for i in y:
        y1 = i[lims[0]+lag:lims[1]]
        y2 = i[lims[0]:int(lims[1]-lag)]
        s1, s2, a = 1, 1, 1
        if normalization:
            s1, s2 = sdev(y1), sdev(y2)
        if average:
            a = len(y1)
        c.append(stats.pearsonr(y1, y2)[0]/(s1*s2*a))

    if dims <= 1:
        c = c[0]

    return c


def crosscorrelation(y1, y2, lag, x=None, lims=None, normalization=False, average=False):
    """
    Calculates the correlation of a signal with a delayed copy of a second
    signal in function of the lag. In other words, measures the similarity
    between two series that are displaced relative to each other.

    :type y: list[float]
    :param y: Data to fit. Single or list of vectors.

    :type x: list[float]
    :param x: x axis.

    :type lag: int
    :param lag: positions shifetd to be analyzed, not in x-axis values.

    :type lims: list[float, float]
    :param lims: limits of the data to autocorrelate. Defauilt is `None` and
        will analyze with all the vector (`[0:len(y)]`).

    :type normalization: boolean
    :param normalization: divides the cross-correlation value by the product of
        the standard deviation of the series in `t` and `t+lag`.

    :type average: boolean
    :param average: divides the cross-correlation value by the length of the
        signals.

    :returns: autocorrelation values, either in a list or single value.
    :rtype: float, list[float]
    """
    if lims is None:
        lims=[0, len(y1)]

    y1 = y1[lims[0]+lag:lims[1]]
    y2 = y2[lims[0]:lims[1]-lag]

    s1, s2, a = 1, 1, 1
    if normalization:
        s1, s2 = sdev(y1), sdev(y2)

    if average:
        a = len(y1)

    return np.dot(y1, y2)/(s1*s2*a)


def derivative(y, x=None, s=1, deg=1):
    """
    Calculates the derivative function of a vector. It does so by calculating
    the slope on a point using the position of the neighboring points.

    :type y: list[float]
    :param y: Data to fit. Single or list of vectors.

    :type x: list[float]
    :param x: x axis. If `None`, an axis will be created from 1..N, where N is
        the length of `y`. Default is `None.

    :type s: int
    :param s: size of the range to each side of the point to use to calculate
        the slope. Default is 1.

    :type deg: int
    :param deg: degree of the derivative. That is, number of timess the vector
        will be derivated. In other words, if `deg=2`, will calculate the
        second derivative. Default is `1`.

    :returns: derivative values, either in a list or single value.
    :rtype: float, list[float]
    """

    y = copy.deepcopy(y)


    dims = len(np.array(y).shape)
    if dims == 1:
        y = [y]

    tl = len(y[0])

    if x is None:
        x = list(range(tl))

    for _ in range(deg):
        fd = []
        for j in y:
            temp = [((j[i + s] - j[i]) / (x[i + s] - x[i])) if i <= s else
                    ((j[i] - j[i - s]) / (x[i] - x[i - s])) if i >= tl - s else
                    ((j[i + s] - j[i - s]) / (x[i + s] - x[i - s])) for i in range(tl)]
            fd.append(temp)
        y = fd

    if dims == 1:
        fd = fd[0]

    return fd


def peaksimilarity(y1, y2, p1, p2, n=5, x=None, plot=False, cmm='inferno',
                   fontsize=10, title='Peak similarity'):
    """
    Calculates the similarity matrix as described in ´doi:10.1142/S021972001350011X´.
    It quantifyes the difference of the derivative function of the peaks and
    puts them into a matrix. This can be used for peak alignment.

    :type y1: list[float]
    :param y1: First vector to compare.

    :type y2: list[float]
    :param y2: Second vector to compare.

    :type p1: list[float]
    :param p1: List of peaks to compare from `y1`. It can be different in positions
        and length than `p2`.

    :type p2: list[float]
    :param p2: List of peaks to compare from `y2`. It can be different in positions
        and length than `p1`.

    :type x: list[float]
    :param x: x axis. If ´None´, an axis will be created from 1..N, where N is
        the length of ´y1´. Default is `None.

    :type n: int
    :param n: size of the range to each side of the point to use to calculate
        the slope. Default is 5.

    :type plot: boolean
    :param plot: If true, plots the similarity matrix. Default is ´False´.

    :type cmm: string
    :param cmm: Color map for the plot. Default is ´inferno´.

    :type fontsize: int
    :param fontsize: Size of the font in the plot. Default is ´10´.

    :type title: string
    :param title: Title of the plot. Default is ´´.

    :returns: similarity matrix in the form of a 2-d list.
    :rtype: list[float]
    """
    if x is None:
        x = [i for i in range(len(y1))]

    if len(p1) < len(p2):
        p1.append(0)
    elif len(p1) > len(p2):
        p2.append(0)

    peaks = [p1, p2]

    peaks = valtoind(peaks, x)

    sm = []
    names = []
    for i in peaks[0]:
        temp = []
        names.append(i)
        for j in peaks[1]:
            s = 0
            for k in range(n*2+1):
                s += 1-abs(y1[i-k]-y2[j-k])/(2*max((abs(y1[i-k]), abs(y2[j-k]))))
            temp.append(s/(n*2+1))
        sm.append(temp)

    if plot:
        fig = plt.figure(tight_layout=True, figsize=(6, 7.5))
        plt.rc('font', size=fontsize)
        ax = fig.add_subplot()
        ax.imshow(sm, cmap=cmm)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        p0 = [x[int(i)] for i in peaks[0]]
        p1 = [x[int(i)] for i in peaks[1]]
        ax.set_xticklabels(p0)
        ax.set_yticklabels(p1)
        plt.ylabel('y2')
        plt.xlabel('y1')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.show()

    return sm


def reverse(y):
    """
    Reverses a vector or a list of vectors.

    :type y: list[float]
    :param y: list of vectors to reverse. Single or multiple.

    :rtype: list[float]
    :returns: Reversed data.
    """
    dims = len(np.array(y).shape)

    if dims == 1:
        y = [y]

    rev = [list(reversed(row)) for row in y]

    if dims == 1:
        rev = rev[0]

    return rev


def count(y, value=0):
    """
    Counts the number of values that coincide with `value`.

    :type y: list[float]
    :param y: vector or list of vectors to search for values.

    :type value: float
    :param value: Value or list of values to search in `y`. If many values are
        passed, then it returns a list with the counted equalities. Default is 0.

    :rtype: list[float]
    :returns: Reversed data.
    """
    y = copy.deepcopy(y)
    dims_val = len(np.array(value).shape)
    dims_y = len(np.array(y).shape)

    if dims_val < 1:
        value = [value]

    if dims_y == 1:
        y = [y]

    count = [[list(row).count(val) for val in value] for row in y]

    if dims_val < 1:
        count = [i[0] for i in count]

    if dims_y == 1:
        count = count[0]

    return count


def vectortoimg(y, negatives='remove', inverted=False):
    """
    Transforms 1-D vecctors into an image. Useful for more insightful results
        when performing deep learning.

    :type y: list[float]
    :param y: vector or list of vectors.

    :type negatives: str
    :param negatives: Specify what to do with negative values. If `remove` then negative values are set to `0`. If
        `globalmin` then all the data is shifted up so the global minimum is `0`. Default it `remove`.

    :type inverted: boolean
    :param inverted: If `True`, inverts the process to take an image to a vector. Default is `False`.

    :rtype: list[list[list[int]]]
    :returns: 3-D list containing 0 or 1.
    """
    y = copy.deepcopy(y)
    y_dims = len(np.array(y).shape)

    if y_dims == 1:
        y = [y]

    gm = False
    if negatives == 'globalmin':
        gm = True

    y = normtoglobalmax(y, globalmin=gm)
    y_len = len(y[0])
    imgs = [[[0 for _ in range(y_len)] for _ in range(y_len)] for _ in y]
    for i in range(len(y)):
        for j in range(y_len):
            if negatives == 'remove' and y[i][j] < 0:
                y[i][j] = 0.00
            val = int(round(y[i][j]*(y_len-1), 0))
            imgs[i][int(val)][j] = 1.00

    if y_dims == 1:
        imgs = imgs[0]

    return imgs


def deconvolution(y, x, pos, method='gauss', shift=5, look=None, pp=False):
    """
    BETA: Deconvolutes a spectra into a defined number of distributions. This will
        fit dsitributions on the declared positions `pos` of peaks and change
        their shape acoording to the difference betwwen the spectra and the
        sum (convoilution) of the fittings.

    :type y: list[float]
    :param y: vector to deconvolute.

    :type x: list[float]
    :param x: axis.

    :type pos: list[float]
    :param pos: positions of peaks in x values

    :type method: string
    :param method: The selected shape of the fittings. Options include: `gauss`,
        `lorentz`, and `voigt`.

    :type pp: boolean
    :param pp: If `True`, it prints parameters of the fittings. Default
         is `False`.

    :rtype: list[list[float]]
    :returns: 2-D list of the fittings.
    """
    if look is None:
        look = [0, len(y)]
    else:
        look = valtoind(look, x)

    if len(np.array(pos).shape) == 0:
        pos = [pos]

    n = len(pos)
    gamma, sigma, alpha = [1 for _ in pos], [1 for _ in pos], [1 for _ in pos]

    ax = x # x becomes the optimization variables

    def objective(x):

        fit = []
        for i in range(n):
            if method == 'voigt':
                fit.append(voigtfit(x=ax, pos=x[i], gamma=x[n+i], sigma=x[2*n+i], alpha=x[3*n+i], manual=True))
            if method == 'lorentz':
                fit.append(lorentzfit(x=ax, pos=x[i], gamma=x[i+n], alpha=x[2*n+i], manual=True))
            if method == 'gauss':
                fit.append(gaussfit(x=ax, pos=x[i], sigma=x[i+n], alpha=x[2*n+i], manual=True))

        error = 0

        for i in range(look[0], look[1]):
            s = 0
            for j in range(len(fit)):
                s += fit[j][i]

            error += (y[i]-s)**2

        return error**0.5

    def constraint1(x):
        # is_zero=0
        # fit = []
        # for i in range(n):
        #     # fit.append(spep.voigtfit(x=ax, pos=pos[i], gamma=x[i], sigma=x[n+i], alpha=x[2*n+i], manual=True))
        #     # fit.append(spep.lorentzfit(x=ax, pos=pos[i], gamma=x[i], alpha=x[n+i], manual=True))
        #     fit.append(spep.gaussfit(x=ax, pos=x[i], sigma=x[i+n], alpha=x[2*n+i], manual=True))

        # for i in range(look[0], look[1]):
        #     s = 0
        #     for j in range(len(fit)):
        #         s += fit[j][i]
        #     return y[i]-s
        return 0


    initial = []
    bnds = []
    for i in range(n):
        bnds.append([pos[i]-shift, pos[i]+shift])
        initial.append(pos[i])
    if method == 'voigt':
        for i in range(n):
            bnds.append([0.001, 999])
            initial.append(gamma[i])
    for i in range(n):
        bnds.append([0.001, 999])
        initial.append(sigma[i])
    for i in range(n):
        bnds.append([0.001, max(y)*99999])
        initial.append(alpha[i])

    x0 = np.array(initial, dtype='object')  # master vector to optimize, initial values
    con1 = {'type': 'ineq', 'fun': constraint1}
    cons = ([con1])
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
    x = solution.x

    if pp:
        print(x)

    y0 = [0 for _ in ax]
    f = []
    for i in range(n):
        if method == 'gauss':
            temp = gaussfit(x=ax, pos=x[i], sigma=x[i+n], alpha=x[i+2*n], manual=True)
        if method == 'lorentz':
            temp = lorentzfit(x=ax, pos=x[i], gamma=x[i+n], alpha=x[i+2*n], manual=True)
        if method == 'voigt':
            temp = voigtfit(x=ax, pos=x[i], gamma=x[i+n], sigma=x[i+2*n], alpha=x[i+3*n], manual=True)
        f.append(temp)
        for j in range(len(ax)):
            y0[j] += temp[j]

    return f[0]


def intersections(y1, y2):
    """
    Find approximate intersection points of two curves.

    :type y1: list[float]
    :param y1: first curve.

    :type y2: list[float]
    :param y2: second curve.

    :rtype: list[list[float]]
    :returns: coordinates of aproiximate intersecctions.
    """

    intersections = []

    # Ensure both curves have the same length.
    assert len(y1) == len(y2), "Both curves should have the same length."

    for i in range(len(y1) - 1):
        # Check if y1[i] is below y2[i] and y1[i+1] is above y2[i+1] (indicating a crossing)
        # or if y1[i] is above y2[i] and y1[i+1] is below y2[i+1] (indicating another crossing)
        if (y1[i] < y2[i] and y1[i+1] > y2[i+1]) or (y1[i] > y2[i] and y1[i+1] < y2[i+1]):
            intersections.append(i)  # Record the x-value where the intersection occurs.

    return intersections
