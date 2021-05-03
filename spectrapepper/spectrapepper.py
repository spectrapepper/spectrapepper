"""Main module that contains all the functions, since this package is
function-based for simplicity. Keep in mind that some of this functions are
only ment to be used inside this file, however they all might be useful in
other code.
"""

import math
import random
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import interpolate
from scipy.stats import stats
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import scipy.interpolate as si
from scipy.sparse.csgraph import _validation
from scipy.sparse.linalg import spsolve
import itertools
from itertools import combinations
import statistics as sta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

"""""""""""""""""""""""""""""""""""""""
          LOAD HEADERS
"""""""""""""""""""""""""""""""""""""""


def load_headers(file):
    def check_header(file):
        with open(file) as f:
            first = f.read(1)
        return first not in '.-0123456789'
    header = []
    new_data = []
    raw_data = open(file, "r")
    for row in raw_data:
        if check_header(file) and len(header) == 0:
            header = row
            header = header.split()
        else:
            row = row.replace(",", ".")
            s_row = str.split(row)
            s_row = np.array(s_row, dtype=float)
            new_data.append(list(s_row))
    raw_data.close()
    return new_data, header


def load_data(file):
    """Load data from a standard text file obtaned from LabSpec and other
    spectroscopy instruments. When single measurement these come in columns with
    the first one being the x-axis. When it is a mapping, the first row is the
    x-axis and the following are the measruements. Sometimes the first 2 columns
    will be the coordinates. No headers, yet.

    Parameters
    ----------
    file : TYPE
        Location of data file.

    Returns
    -------
    new_data : TYPE
        Lodaded data.

    """
    new_data = []
    raw_data = open(file, "r")
    for row in raw_data:
        row = row.replace(",", ".")
        row = row.replace(";", " ")
        row = row.replace("NaN", "")
        row = row.replace("nan", "")
        s_row = str.split(row)
        s_row = np.array(s_row, dtype=np.float)
        new_data.append(s_row)
    raw_data.close()
    return new_data


def butter_lowpass_filter(data, cutoff=0.25, fs=30, order=2, nyq=0.75 * 30):
    """Butter low pass filter for a single or spectra or a list of them.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    cutoff : TYPE, optional
        Desired cutoff frequency of the filter. The default is 0.25.
    fs : TYPE, optional
        Sample rate in Hz . The default is 30.
    order : TYPE, optional
        Sin wave can be approx represented as quadratic. The default is 2.
    nyq : TYPE, optional
        Nyquist frequency, 0.75*fs is a good value to start. The default is 0.75*30.

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    """
    y = list(data)  # so it does not change the input list
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    dim = np.array(y)
    if len(np.array(y).shape) > 1:
        for i in range(len(y)):
            y[i] = filtfilt(b, a, y[i])
    else:
        y = filtfilt(b, a, y)
    return y


def normtomax(data):
    """Normalizes spectras to the maximum value of each, in other words, the
    maximum value of each spectras is set to 1.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    """
    y = list(data)  # so it does not chamge the input list
    dims = len(np.array(data).shape)  # detect dimensions
    if dims >= 2:
        for i in range(len(y)):
            max_data = max(y[i])
            for j in range(len(y[i])):
                y[i][j] = y[i][j] / max_data
    else:
        max_data = max(y)
        for i in range(len(y)):
            y[i] = y[i] / max_data
    return y


def normtovalue(data, val):
    """Normalizes the spectras to a set value, in other words, the define value
    will be reescaled to 1 in all the spectras.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    val : float
        Value to normalize to.

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    """
    y = list(data)  # so it does not chamge the input list
    dims = len(np.array(data).shape)  # detect dimensions
    if dims >= 2:
        for i in range(len(y)):  # for all spectras
            for j in range(len(y[i])):  # for all elements in the spectra
                y[i][j] = y[i][j] / val  # divide by the maximum
    else:
        for i in range(len(y)):  # for all spectras
            y[i] = y[i] / val
    return y


def baseline_als(y, lam=100, p=0.001, niter=10):
    """Calculation of the baseline using Asymmetric Least Squares Smoothing. This
    script only makes the calculation but it does not remove it. Original idea of
    this algorythm by P. Eilers and H. Boelens (2005), and available details at:
    https://stackoverflow.com/a/29185844/2898619

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    lam : TYPE, optional
        Also known as lambda, smoothness. The default is 100.
    p : TYPE, optional
        Asymetry. The default is 0.001.
    niter : TYPE, optional
        Niter. The default is 10.

    Returns
    -------
    z : TYPE
        DESCRIPTION.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def no_baseline(data, lam=100000, p=0.001, niter=10):
    """This takes baseline_als and removes it from the spectra.

    Keyword arguments:
    data -- single spectra
    lam -- also known as lambda, smoothness
    p -- asymetry
    niter -- niter
    """
    y = list(data)
    for i in range(len(y)):
        y[i] = y[i] - baseline_als(y[i], lam, p, niter)
    return y


def bspline(cv, n=100, degree=3, periodic=False):
    """Removes the baseline using b-spline. Find usful details and guidelines in
    https://stackoverflow.com/a/34807513/2898619

    Parameters
    ----------
    cv : TYPE
        list of spectras.
    n : TYPE, optional
        number of samples to return. The default is 100.
    degree : TYPE, optional
        curve (polynomial) degree. The default is 3.
    periodic : TYPE, optional
        rue if curve is closed, False if curve is open. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    cv = np.asarray(cv)
    count = len(cv)
    # open, prevent degree from exceeding count-1
    degree = np.clip(degree, 1, count - 1)
    kv = None  # Calculate knot vector
    kv = np.concatenate(([0] * degree,
                         np.arange(count - degree + 1),
                         [count - degree] * degree))
    u = np.linspace(periodic, (count - degree), n)  # Calculate query range
    return np.array(si.splev(u, (kv, cv.T, degree))).T  # Calculate result


def lorentzian_fit(data_x, ax, wid=4, it=100):
    """Fit a Lorentz distributed curve for a single spectra. Good guidelines
    for similar structures can be found at http://emilygraceripka.com/blog/16.

    Parameters
    ----------
    data_x : TYPE
        single spectra.
    ax : TYPE
        x-axis.
    wid : TYPE, optional
        fitted curve width. The default is 4.
    it : TYPE, optional
        number of itereations. The default is 100.

    Returns
    -------
    fit : TYPE
        DESCRIPTION.
    """
    x = list(ax)  # change the variable so it doesnt change the original
    y = list(data_x)  # change the variable so it doesnt change the original
    cen = 0  # position (wavelength, x value)
    peak_i_pos = 0  # position (axis position, i value)
    amp = max(y)  # maximum value of function, peak value, amplitud of the fit
    fit = []  # fit function
    err = []  # error log to choose the optimum
    for i in range(len(y)):  # for all the data
        if amp == y[i]:  # if it is the maximum
            cen = ax[i]  # save the axis value x
            peak_i_pos = int(i)  # also save the position i
            break
    for i in range(it):  # search "it" iterations
        temp = 0  # reset temporal error
        fit = []  # reset lorentz fit array
        # pos of the peak respect to the reference measured peak
        p = cen - 0.5 + (0.01 * i)
        for j in range(len(x)):  # for all the points
            fit.append(amp * wid**2 / ((x[j] - p)**2 + wid**2))
        for j in range(
                peak_i_pos -
                25,
                peak_i_pos +
                25):  # error between -25 and +25 of the peak position
            temp = temp + (fit[j] - y[j]) ** 2  # total error
        err.append(temp)  # log error
    fit = []  # reset array
    for i in range(len(err)):  # look for the minimum error
        if min(err) == err[i]:  # if it is the minimum error
            p = cen - 0.5 + (0.01 * i)  # then calculate the fit again
            for j in range(len(y)):
                fit.append(amp * wid**2 / ((x[j] - p)**2 + wid**2))
            break
    return fit


def gaussian_fit(data_x, ax, sigma=4.4, it=100):
    """Fit a Gaussian distributed curve for a single spectra. Good guidelines
    for similar structures can be found at http://emilygraceripka.com/blog/16.


    Parameters
    ----------
    data_x : TYPE
        single spectra.
    ax : TYPE
        x-axis.
    sigma : TYPE, optional
        sigma parameter of distribution. The default is 4.4.
    it : TYPE, optional
        number of itereations. The default is 100.

    Returns
    -------
    fit : TYPE
        DESCRIPTION.
    """
    x = list(ax)  # change the variable so it doesnt change the original
    y = list(data_x)  # change the variable so it doesnt change the original
    cen = 0  # position (wavelength, x value)
    peak_i_pos = 0  # position (axis position, i value)
    amp = max(y)  # maximum value of function, peak value,a mplitud of the fit
    fit = []  # fit function
    err = []  # error log to choose the optimum
    for i in range(len(y)):  # for all the data
        if amp == y[i]:  # if it is the maximum
            cen = ax[i]  # save the axis value x
            peak_i_pos = int(i)  # also save the position i
            break
    for i in range(it):  # search "it" iterations
        temp = 0  # reset temporal error
        fit = []  # reset lorentz fit array
        # pos of the peak respect to the reference measured peak
        p = cen - 0.5 + (0.01 * i)
        for j in range(len(x)):  # for all the points
            fit.append((11 * amp) * (1 / (sigma * (np.sqrt(2 * np.pi))))
                       * (np.exp(-0.5 * (((x[j] - p) / sigma)**2))))
        for j in range(
                peak_i_pos -
                25,
                peak_i_pos +
                25):  # error between -25 and +25 of the peak position
            temp = temp + (fit[j] - y[j]) ** 2  # total error
        err.append(temp)  # log error
    fit = []  # reset array
    for i in range(len(err)):  # look for the minimum error
        if min(err) == err[i]:  # if it is the minimum error
            p = cen - 0.5 + (0.01 * i)  # then calculta the fit again
            for j in range(len(y)):
                fit.append((11 * amp) * (1 / (sigma * (np.sqrt(2 * np.pi))))
                           * (np.exp(-0.5 * (((x[j] - p) / sigma)**2))))
            break
    return fit


def cortopos(vals, ax):
    """To translate values to a position in an axis, basically searching for
    the position of  avalue. Normally they dont fit perfectly, so it is useful
    to use this tool that aproximates to the closest.

    Parameters
    ----------
    vals : TYPE
        List of values to be searched and translated.
    ax : TYPE
        Axis.

    Returns
    -------
    pos : TYPE
        Postions.
    """
    axis = list(ax)  # axis
    y = list(vals)  # axis values that you want to transalte to position

    if len(np.array(y).shape) > 1:
        pos = [[0 for _ in range(len(y[0]))] for _ in range(
            len(y))]  # i position of area limits
        for i in range(
                len(y)):  # this loop takes the aprox x and takes its position
            for j in range(len(y[0])):
                dif_temp = 999  # safe initial difference
                temp_pos = 0  # temporal best position
                for k in range(len(axis)):  # search in x_axis
                    if abs(y[i][j] - axis[k]) < dif_temp:  # compare if better
                        temp_pos = k  # save best value
                        dif_temp = abs(y[i][j] - axis[k])  # calculate new diff
                y[i][j] = axis[temp_pos]  # save real value in axis
                pos[i][j] = temp_pos  # save the position
    else:
        pos = []  # i position of area limits
        for i in range(
                len(y)):  # this loop takes the aprox x and takes its position
            dif_temp = 999  # safe initial difference
            temp_pos = 0  # temporal best position
            for k in range(len(axis)):
                if abs(y[i] - axis[k]) < dif_temp:
                    temp_pos = k
                    dif_temp = abs(y[i] - axis[k])
            y[i] = axis[temp_pos]  # save real value in axis
            pos.append(temp_pos)  # save the position
    return pos


def areacalculator(x_data, limits):
    """Area calculator using the data (x_data) and the limits in position, not
    values.

    Parameters
    ----------
    x_data : TYPE
        DESCRIPTION.
    limits : TYPE
        DESCRIPTION.

    Returns
    -------
    areas : TYPE
        DESCRIPTION.
    """
    data = list(x_data)  # axis
    lims = list(limits)  # axis values that you want to transalte to position
    areas = [[0 for _ in range(len(lims))]
             for _ in range(len(data))]  # final values of areas
    for i in range(len(data)):  # calculate the areas for all the points
        for j in range(len(lims)):  # for all the areas
            # calculate the sum
            areas[i][j] = np.sum(data[i][lims[j][0]:lims[j][1]])
    return areas


def bincombs(n, s_min, s_max):
    """Returns all possible unique combinations.

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    s_min : TYPE
        DESCRIPTION.
    s_max : TYPE
        DESCRIPTION.

    Returns
    -------
    iters : TYPE
        DESCRIPTION.
    """
    e = 0  # counter
    iters = []  # final matrix
    temp = []  # temp matrix
    stuff = []  # matrix of ALL combinatios

    for i in range(n):  # create possibilities vector (0 and 1)
        stuff.append(1)
        stuff.append(0)

    for subset in itertools.combinations(stuff, n):
        temp.append(subset)  # get ALL combinations possible from "stuff"

    for i in range(len(temp)):  # for all the possibilities...
        e = 0
        for j in range(len(iters)):  # check if it already exists
            if temp[i] == iters[j]:  # if it maches, add 1
                e += 1
            else:
                e += 0  # if not, add 0
        if e == 0 and sum(temp[i]) >= s_min and sum(temp[i]) <= s_max:
            # if new and fall into the criteria, then add
            iters.append(temp[i])

    return iters


def normsum(data):
    """Normalizes the sum under the curve to 1, for single or multiple spectras.

    Parameters
    ----------
    data : TYPE
        Single spectra or a list of them.

    Returns
    -------
    y : TYPE
        Processed data.
    """
    y = list(data)
    dims = len(np.array(data).shape)
    if dims >= 2:
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
    """Normalizes a list of spectras to the global max.

    Parameters
    ----------
    data : TYPE
        List of spectras.

    Returns
    -------
    y : List
        Data normalized to the global max.
    """
    y = list(data)
    dims = len(np.array(data).shape)
    if dims >= 2:
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


def interpolation(files_names):
    """Returns an array with the first item being the interpolated
    data and the second item being the axis.

    Parameters
    ----------
    files_names : TYPE
        List of locations of the data to interpolate (should change).

    Returns
    -------
    master_y : TYPE
        Interpolated data.
    x_new : TYPE
        New axis.

    """
    ### LOAD DATA ###
    master_x = []
    temp_y = []
    files = list(files_names)
    for i in range(len(files)):
        data_temp = load_data(files[i])
        master_x.append(list(data_temp[0]))
        y = list(np.delete(data_temp, 0, 0))  # delete axis from
        y = np.delete(y, 0, 1)  # delete first column
        y = np.delete(y, 0, 1)  # delete second (first) column
        temp_y.append(y)
    ### END OF LOAD DATA ###

    ### NEW AXIS ###
    new_step = 1
    new_start = -1
    new_end = 99999999
    for i in range(len(master_x)):
        if min(master_x[i]) > new_start:
            new_start = math.ceil(min(master_x[i]))
        if max(master_x[i]) < new_end:
            new_end = math.floor(max(master_x[i]))
    x_new = np.arange(new_start, new_end + new_step, new_step)
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


def grau(data):
    """This function evaluates the MSE in 3 dimensions (x,y,z) for a set of
    data vectors.

    Parameters
    ----------
    data : TYPE
        A list of lists (no typo) of variables to compare.

    Returns
    -------
    merged : TYPE
        A list with each combination and the R2 score obtained.
    """
    data = list(data)
    tup = [i for i in range(len(data))]  # number list of data type sets
    combs = tuple(combinations(tup, 3))  # make all combinations

    R2 = []  # R2 list
    for i in range(len(combs)):  # for all the combinations
        xs = data[combs[i][0] - 1]  # temporal x axis
        ys = data[combs[i][1] - 1]  # temporal y axis
        zs = data[combs[i][2] - 1]  # temporal z axis
        A = []  # fit function parameter
        b = []  # fit function parameter
        for j in range(len(data[0])):  # for all the data points
            A.append([xs[j], ys[j], 1])  # A value
            b.append(zs[j])  # b value
        b = np.matrix(b).T  # transpose amtrix form
        A = np.matrix(A)  # matrix form
        fit = (A.T * A).I * A.T * b  # evaluate fir
        rss = 0  # residual sum of squares
        tss = 0  # total sum of squares
        for i in range(len(data[0])):  # calculate mse for all the points
            rss += (zs[i] - (xs[i] * fit[0] + ys[i] *
                    fit[1] + fit[2]))**2  # residual sum
            tss += (zs[i] - np.mean(zs))**2  # total error
        R2.append(round(float(1 - rss / tss), 2))  # R2
    merged = np.c_[combs, R2]  # merge to sort
    return merged


def groupscores(all_targets, used_targets, predicted_targets):
    """Calculates the individual scores for a ML algorythm (i.e.: LDA, PCA, etc).

    Parameters
    ----------
    all_targets : TYPE
        List of all real targets (making sure all groups are here).
    used_targets : TYPE
        Targets to score on.
    predicted_targets : TYPE
        Prediction of used_targets.

    Returns
    -------
    g_scores : TYPE
        List of scores.
    """
    g_count = [0 for i in range(
        int(max(all_targets) + 1))]  # list to count points per group
    # list to store the scores
    g_scores = [0 for i in range(int(max(all_targets) + 1))]
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
    """Calculates the distance between points and the center of mass (CM) of
    clusters and sets the predictoin to the closest CM. This score may be
    higher or lower than the algorithm score.

    Parameters
    ----------
    x_points : list
        Coordinates of x-axis.
    y_points : list
        Coordinates of y-axis.
    target : list
        Targets of each point.

    Returns
    -------
    score : float
        Score by comparing CM distances.
    p : list
        Prediction using CM distances.
    a : list
        X-axis coordinates of ths CMs.
    b : list
        Y-axis coordinates of the Cms.
    """
    x_p = list(x_points)
    y_p = list(y_points)
    tar = list(target)
    g_n = int(max(target) + 1)

    a = [0 for i in range(g_n)]  # avg D1
    b = [0 for i in range(g_n)]  # avg D2
    c = [0 for i in range(g_n)]  # N for each group
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
            temp3.append(((x_p[i] - a[j])**2
                          + (y_p[i] - b[j])**2)**0.5)

            if temp3[j] < temp2:
                temp2 = temp3[j]
                temp1 = j

        p.append(temp1)
        d.append(temp3)

        if tar[i] == temp1:
            correct += 1

    score = round(correct / len(tar), 2)

    return score, p, a, b


def mdscore(x_points, y_points, target):
    """Calculates the distance between points and the median center (MD) of
    clusters and sets the predictoin to the closest MD. This score may be
    higher or lower than the algorithm score.

    Parameters
    ----------
    x_points : TYPE
        DESCRIPTION.
    y_points : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    score : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    """
    x_p = list(x_points)
    y_p = list(y_points)
    g_n = max(target) + 1
    tar = list(target)

    a = [0 for i in range(g_n)]
    b = [0 for i in range(g_n)]
    d = []  # distances
    p = []  # predictions

    x_s = [[] for i in range(g_n)]
    y_s = [[] for i in range(g_n)]

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
            temp3.append(((x_p[i] - a[j])**2
                          + (y_p[i] - b[j])**2)**0.5)

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
    """Normalizes the spectras to a particular peak.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    x_axis : TYPE
        DESCRIPTION.
    peak : TYPE
        DESCRIPTION.
    shift : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    data = list(data)
    x_axis = list(x_axis)
    peak = [int(peak)]
    shift = int(shift)

    pos = cortopos(peak, x_axis)
    section = data[pos[0] - shift:pos[0] + shift]
    highest = peakfinder(section, l=int(shift / 2))

    c = 0
    for i in range(len(highest)):
        if highest[i] == 1:
            c = i
            break

    local_max = data[pos[0] - shift + c]
    data = data / local_max
    return data


def peakfinder(data, l=10):
    """

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    l : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    is_max : TYPE
        DESCRIPTION.
    """
    y = list(data)
    look = l  # positions to each side to look max or min (2*look)
    is_max = [0 for i in data]
    is_min = [0 for i in data]
    for i in range(look, len(y) - look):  # start at "look" to avoid o.o.r
        lower = 0  # negative if lower, positive if higher
        higher = 0
        for j in range(look):
            if (y[i] <= y[i - look + j] and
                    y[i] <= y[i + j]):  # search all range lower
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


def confusionmatrix(tt, tp, gn):
    """

    Parameters
    ----------
    tt : TYPE
        DESCRIPTION.
    tp : TYPE
        DESCRIPTION.
    gn : TYPE
        DESCRIPTION.

    Returns
    -------
    m : TYPE
        DESCRIPTION.
    """
    tt = list(tt)  # training targets
    tp = list(tp)  # predicted targets
    gn = int(gn)  # group number
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
    return m


def avg(data):
    """

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    avg_data : TYPE
        DESCRIPTION.
    """
    data = list(data)
    avg_data = np.array([0 for i in range(len(data[0]))])
    for i in data:
        avg_data = avg_data + i
    avg_data = avg_data / len(data)
    return avg_data


def sdev(data):
    """

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    curve_std : TYPE
        DESCRIPTION.

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


def peakfit(data, ax, pos, look=20, shift=5):
    """Feats peak as an optimization problem.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    look : TYPE, optional
        DESCRIPTION. The default is 20.
    shift : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    # initial guesses
    ax = list(ax)
    y = list(data)
    look = int(look)
    s = int(shift)
    pos = int(pos)

    amp = max(y) / 2
    wid = 5

    def objective(x):
        fit = []
        error = 0
        for j in range(len(ax)):  # for all the points
            fit.append(x[0] * x[1]**2 / ((ax[j] - ax[pos])**2 + x[1]**2))
        for j in range(
                pos -
                look,
                pos +
                look):  # error between +-look of the peak position
            error += (fit[j] - y[j])**2  # total error
        return error

    def constraint1(x):
        return 0

    for i in range(pos - s, pos + s):
        if y[i] > y[pos]:
            pos = i

    x0 = np.array([amp, wid])  # master vector to optimize, initial values
    bnds = [[0, max(y)], [0.1, 100]]
    con1 = {'type': 'ineq', 'fun': constraint1}
    cons = ([con1])
    solution = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bnds,
        constraints=cons)
    x = solution.x

    fit = []
    for j in range(len(ax)):  # for all the points
        fit.append(x[0] * x[1]**2 / ((ax[j] - ax[int(pos)])**2 + x[1]**2))
    return fit


def decbound(xx, yy, xlims, ylims, divs=0.01):
    """DESICION BOUNDAEY

    Parameters
    ----------
    xx : TYPE
        DESCRIPTION.
    yy : TYPE
        DESCRIPTION.
    xlims : TYPE
        DESCRIPTION.
    ylims : TYPE
        DESCRIPTION.
    divs : TYPE, optional
        DESCRIPTION. The default is 0.01.

    Returns
    -------
    pmap : TYPE
        DESCRIPTION.
    """
    cmx = list(xx)  # centroids
    cmy = list(yy)
    divs = float(divs)  # step
    map_x = list(np.array(xlims) / divs)  # mapping limits
    map_y = list(np.array(ylims) / divs)

    x_divs = int((map_x[1] - map_x[0]))  # mapping combining 'divs' & 'lims'
    y_divs = int((map_y[1] - map_y[0]))

    # coordinates
    x_cords = [divs * i for i in range(int(min(map_x)), int(max(map_x)))]
    y_cords = [divs * i for i in range(int(min(map_y)), int(max(map_y)))]

    pmap = [[0 for i in range(x_divs)] for j in range(y_divs)]  # matrix

    for i in range(len(x_cords)):
        for j in range(len(y_cords)):
            t1 = 9999999999999999999999999
            #grad = [] #####
            for k in range(len(cmx)):
                d = (cmx[k] - x_cords[i])**2 + (cmy[k] - y_cords[j])**2
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
    """Performs an N dimentional regression.

    Parameters
    ----------
    target : TYPE
        Y-axis values.
    variable : TYPE
        X-axis values.
    cov : TYPE, optional
        If 1 is regression with covariance, like spearman. The default is 0.
    exp : TYPE, optional
        For exponent regressions. Not working yet. The default is 0.

    Returns
    -------
    prediction : TYPE
        Prediction of the fitting.
    fit : TYPE
        Parameters of the regression.
    """
    target = list(target)
    master = list(variable)
    cov = int(cov)
    exp = int(exp)  # for non-linear fittings in the future

    if cov == 1:
        master.append(target)  # array of arrays
        print(len(master))
        pos = [i for i in range(len(master[0]))]  # ascending values
        df = pd.DataFrame(data=master[0], columns=['0'])  # 1st col is 1st var
        for i in range(len(master)):  # for all variables and target
            df[str(2 * i)] = master[i]  # insert into dataframe
            df = df.sort_values(by=[str(2 * i)])  # sort ascending
            df[str(1 + 2 * i)] = pos  # translate to position
            df = df.sort_index()  # reorder to maintain original position

        master = [df[str(2 * i + 1)]
                  for i in range(int(len(df.columns) / 2 - 1))]
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

    Parameters
    ----------
    lims : TYPE
        DESCRIPTION.
    x_points : TYPE
        DESCRIPTION.
    y_points : TYPE
        DESCRIPTION.
    groups : TYPE
        DESCRIPTION.
    divs : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    master : TYPE
        DESCRIPTION.
    """
    divs = float(divs)  # resolution, step

    map_x = list(np.array(lims[:2]) / divs)  # mapping limits
    map_y = list(np.array(lims[2:]) / divs)

    x_p = list(x_points)  # points coordinates
    y_p = list(y_points)

    groups = list(groups)  # target (group) list
    n_groups = int(max(groups) + 1)  # number of groups

    x_divs = int((map_x[1] - map_x[0]))  # divisons for mapping
    y_divs = int((map_y[1] - map_y[0]))

    # coordinates
    x_cords = [divs * i for i in range(int(min(map_x)), int(max(map_x)))]
    y_cords = [divs * i for i in range(int(min(map_y)), int(max(map_y)))]

    master = []  # to store the maps for each group

    for l in range(n_groups):
        pmap = [[0 for i in range(x_divs)]
                for j in range(y_divs)]  # individual matrices
        for i in range(len(x_cords) - 1):
            for j in range(len(y_cords) - 1):
                count = 0
                for k in range(len(x_p)):
                    if (x_cords[i] < x_p[k] and x_p[k] < x_cords[i + 1] and
                        y_cords[j] < y_p[k] and y_p[k] < y_cords[j + 1] and
                            groups[k] == l):
                        count += 1
                pmap[j][i] = count

        maximum = max(np.array(pmap).flatten())
        pmap = np.array(pmap) / maximum
        master.append(pmap)
    return master


def colormap(c, name='my_name', n=100):
    """Just to simplify a bit the creatin of colormaps.

    Parameters
    ----------
    c : list
        List of colors thaat you ant on the colormap
    name : string, optional
        Name of the colormap. The default is 'my_name'.
    n : TYPE, optional
        Divisions. The default is 100.

    Returns
    -------
    cmap : TYPE
        Colormap in Matplotlib format.
    """
    colors = list(c)  # R -> G -> B
    cmap_name = str(name)
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    return cmap


def isaxis(data):
    """Detects if there is an axis in the data.

    Parameters
    ----------
    data : list
        Data containing spectras an possible axis.

    Returns
    -------
    is_axis : bool
        True if there is axis.
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
    """Deletes columns in a list from start to finish.

    Parameters
    ----------
    data : list
        Data to be trimmed.
    start : int, optional
        Poistion of the starting point. The default is 0.
    finish : int, optional
        Position of the ending point. The default is 0.

    Returns
    -------
    data : list
        Trimed data.
    """
    data = list(data)  # for 1 spectra
    s = int(start)
    f = int(finish)

    if f == 0 or f > len(data):
        f = len(data)

    t = f - s

    for i in range(t):
        data = np.delete(data, s, 1)
    return data


def shuffle(arrays):
    """Merges and shuffles data adn the separates it so it is shuffles together.

    Parameters
    ----------
    arrays : list
        List of arrays of data.

    Returns
    -------
    new_list : list
        List of the shuffled arrays.
    """
    all_list = list(arrays)

    features = all_list[0]
    for i in range(1, len(all_list)):
        features = np.c_[features, all_list[i]]

    np.random.shuffle(features)  # shuffle data before training the ML

    new_list = [[] for i in all_list]
    lengths = []

    for i in range(len(all_list)):
        if len(np.array(all_list[i]).shape) >= 2:
            lengths.append(np.array(all_list[i]).shape[1])
        else:
            lengths.append(1)

    for i in range(len(all_list)):
        for j in range(len(all_list[0])):
            if i == 0:
                new_list[i].append(features[j][0:lengths[i]])
            else:
                new_list[i].append(features[j][sum(lengths[0:i])])

    return new_list


def mergedata(data):
    """Merges data, it can merge large vectors. Usuful to merge features
    before performing ML algorythms.

    Parameters
    ----------
    data : TYPE
        List of arrays.

    Returns
    -------
    master : TYPE
        List with the merged data.
    """
    data = list(data)  # list of lists
    master = [[] for i in data[0]]
    for i in range(len(data[0])):
        for j in range(len(data)):
            master[i].extend(data[j][i])
    return master


def logo(lay=90, leng=100, a=1, b=0.8, r1=80, r2=120, lw=2):
    """Prints the logo of spectrapepper

    Parameters
    ----------
    lay : TYPE, optional
        DESCRIPTION. The default is 90.
    leng : TYPE, optional
        DESCRIPTION. The default is 100.
    a : TYPE, optional
        DESCRIPTION. The default is 1.
    b : TYPE, optional
        DESCRIPTION. The default is 0.8.
    r1 : TYPE, optional
        DESCRIPTION. The default is 80.
    r2 : TYPE, optional
        DESCRIPTION. The default is 120.
    lw : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    list
        Prints the logo with matplotlib.
    """
    x = [[1 for i in range(leng)] for j in range(lay)]

    def xl(amp, wid, p):
        return [(amp * wid**2 / ((i - p)**2 + wid**2)) *
                random.randint(99, 101) / 100 for i in range(100)]

    for i in range(len(x)):
        wid = 0.0000002 * (i**4) - 0.0002 * (i**3) + \
            0.0178 * (i**2) - 0.0631 * i + 4.7259
        pos = 0.000001 * (i**4) - 0.0004 * (i**3) + 0.046 * \
            (i**2) - 1.8261 * i + 59.41
        amp = wid * (a)
        x[i] = xl(amp, wid * b, pos)

    axis = [i for i in range(leng)]

    plt.figure(figsize=(10, 14))
    for i in range(len(x)):
        r = [random.randint(r1, r2) / 100 for i in range(leng)]
        curve = np.array(x[i]) + i + r
        plt.plot(curve, color='black', lw=lw)
        plt.fill_between(axis, curve, color='white', alpha=1)
    plt.show()


# data: array of arrays
def shiftref(data, peak_ref=520, mode=1, it=100, plot=True):
    """Shifts the x-axis according to a shift calcualted prior.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    peak_ref : TYPE, optional
        DESCRIPTION. The default is 520.
    mode : TYPE, optional
        DESCRIPTION. The default is 1.
    it : TYPE, optional
        DESCRIPTION. The default is 100.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    peakshift : TYPE
        DESCRIPTION.

    """
    si_data = list(data)
    peak_ref = float(520)
    mode = int(mode)
    it = int(it)
    plot = bool(plot)

    data_trans = []  # transition array
    data_c = []  # curve (y values)
    x_axis = []  # si x axis, later it will change
    fit = []  # fit curves(s), if selected
    shift = []  # axis shift array

    for i in range(len(si_data)):  # conditionning data
        data_trans.append(np.transpose(si_data[i]))  # transpose
        x_axis.append(data_trans[i][0])  # separate axis
        data_c.append(data_trans[i][1])  # separate curve values

    for i in range(len(data_c)):  # depending on the mode chosen...
        if mode == 1:
            # check my_functions for parameters
            fit.append(lorentzian_fit(data_c[i], x_axis[i], 4, it))
        if mode == 2:
            fit.append(gaussian_fit(data_c[i], x_axis[i], 4.4, it))
        if mode == 0:
            fit.append(data_c[i])

    for i in range(len(fit)):  # look for the shift with max value (peak)
        for j in range(len(fit[0])):  # loop in all axis
            if fit[i][j] == max(fit[i]):  # if it is the max value,
                # calculate the diference
                shift.append(x_axis[i][j] - peak_ref)

    temp = 0  # temporal variable
    for i in range(len(shift)):
        temp = temp + shift[i]  # make the average

    peakshift = -temp / len(shift)

    if plot:
        plt.figure()  # figsize = (16,9)
        for i in range(len(data_c)):
            plt.plot(
                x_axis[i],
                data_c[i],
                linewidth=2,
                label='Original' +
                str(i))
            plt.plot(
                x_axis[i],
                fit[i],
                linewidth=2,
                label='Fit' +
                str(i),
                linestyle='--')
        plt.axvline(
            x=peak_ref,
            ymin=0,
            ymax=max(
                data_c[0]),
            linewidth=2,
            color="red",
            label=peak_ref)
        plt.axvline(
            x=peak_ref -
            peakshift,
            ymin=0,
            ymax=max(
                data_c[0]),
            linewidth=2,
            color="yellow",
            label="Meas. Max.")
        plt.gca().set_xlim(peak_ref - 15, peak_ref + 15)
        #plt.gca().set_ylim(0, 2)
        plt.legend(loc=0)
        plt.ylabel('a.u.')
        plt.xlabel('Shift (cm-1)')
        plt.show()

    return peakshift


def classify(data, gnumber=3, glimits=[]):
    """Classifyies targets according to either defined limits or number of groups.
    The latter depends on the defined parameters.

    Parameters
    ----------
    data : TYPE
        Vector with values.
    gnumber : int, optional
        DESCRIPTION. The default is 3 and is the default technique.
    glimits : list, optional
        Defined group limits. The default is [].

    Returns
    -------
    class_targets : TYPE
        Vector with the classificaion from 0 to N.
    group_names : TYPE
        A list with strings with the name of the groups, useful for plotting.
    """
    group_number = int(gnumber)
    group_limits = list(glimits)
    targets = list(data)

    # df_targets['T'] is data from the file, ['NT'] the classification (0 to
    # ...)
    df_targets = pd.DataFrame(data=targets, columns=['T'])
    group_names = []
    class_targets = []  # ['NT']

    # if I set the number of groups
    if group_number > 0:
        group_limits = [[0, 0] for i in range(group_number)]
        df_targets.sort_values(by='T', inplace=True)
        group_size = math.floor(len(targets) / group_number)
        left_over = len(targets) - group_size * group_number
        g_s = []
        for i in range(group_number):
            g_s.append(
                group_size +
                math.floor(
                    (i +
                     1) /
                    (group_number)) *
                left_over)
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
                group_names.append(
                    str(group_limits[i][0]) + ' - ' + str(group_limits[i][1]))
            temp = temp + g_s[i]
        df_targets.sort_index(inplace=True)

    # if I set the limits
    if len(group_limits) >= 1 and group_number <= 1:
        class_targets = [-1 for i in range(len(targets))]

        for i in range(0, len(group_limits)):
            for j in range(len(targets)):

                if targets[j] < group_limits[0]:
                    class_targets[j] = 0

                if targets[j] >= group_limits[len(group_limits) - 1]:
                    class_targets[j] = len(group_limits)

                elif targets[j] >= group_limits[i] and targets[j] < group_limits[i + 1]:
                    class_targets[j] = i + 1

        group_names.append(' < ' + str(group_limits[0]))
        for i in range(0, len(group_limits) - 1):
            group_names.append(
                str(group_limits[i]) + ' - ' + str(group_limits[i + 1]))
        group_names.append(str(max(group_limits)) + ' =< ')

    return class_targets, group_names


def subtractref(
        data,
        ref,
        alpha=0.9,
        sample=0,
        plot=True,
        plot_lim=[
            50,
            200],
        mcr=False):
    """Subtracts a reference spectra (i.e.: air) from the measurements. If

    Parameters
    ----------
    data : TYPE
        List of spectras, with x-axis.
    ref : TYPE
        air (reference) data, with x-axis.
    alpha : TYPE, optional
        Manual multiplier. The default is 0.9.
    sample : TYPE, optional
        Sample spectra to work with. The default is 0.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    plot_lim : TYPE, optional
        DESCRIPTION. The default is [50,200].
    mcr : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------gfkl
    data : TYPE
        Data with the subtracetd reference.
    """
    data = list(data)
    air = list(ref)
    sample = int(sample)  # spectrum chosen to work with. 0 is the first
    # multiplication factor to delete the air spectrum on the sample
    alpha = float(alpha)
    plot_lim = list(plot_lim)  # range of the plot you want to see

    x_axis = data[0]  # x axis is first row
    data = np.delete(data, 0, 0)  # delete x_axis from data
    data = list(data)

    if len(air[0]) <= 2:  # if not in correct format
        air = np.transpose(air)  # transpose so it is in rows
    x_axis_air = air[0]  # x axis is first row
    air = air[1]  # make it the same structure as x_axis
    final = data[sample] - air * alpha  # final result

    if plot:
        plt.plot(
            x_axis,
            data[sample],
            linewidth=1,
            label='Original',
            linestyle='--')
        plt.plot(x_axis_air, air, linewidth=1, label='Air', linestyle='--')
        plt.plot(x_axis, final, linewidth=1, label='Final')
        plt.gca().set_xlim(plot_lim[0], plot_lim[1])
        plt.legend(loc=0)
        plt.ylabel('a.u.')
        plt.xlabel('Shift (cm-1)')
        plt.show()

    data.insert(0, x_axis)

    return data


def plotpearson(
        data,
        labels,
        cm="seismic",
        fons=20,
        figs=(
            20,
            17),
    tfs=25,
        ti="Pearson"):
    """Calculates Pearon matrix and returns it in the form of a plot.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    cm : TYPE, optional
        Color map as for matplolib. The default is "seismic".
    fons : TYPE, optional
        Plot font size. The default is 20.
    figs : TYPE, optional
        Plot size. The default is (20,17).
    tfs : TYPE, optional
        Title font size. The default is 25.
    ti : TYPE, optional
        Plot title/name. The default is "pearson".

    Returns
    -------
    Plot.
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
    fig = plt.figure(tight_layout=True, figsize=figsize)

    y = [i + 0.5 for i in range(n)]
    ticker = mpl.ticker.FixedLocator(y)
    formatter = mpl.ticker.FixedFormatter(labels)

    ax = fig.add_subplot(gs[0, 0])
    pcm = ax.pcolormesh(pears, cmap=cm, vmin=-1, vmax=1)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(title, fontsize=titlefs)
    ax.xaxis.set_major_locator(ticker)
    ax.yaxis.set_major_locator(ticker)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(rotation='90')
    plt.show()


"""""""""""""""""""""""""""""""""""""""
      PLOT SPEARMAN
"""""""""""""""""""""""""""""""""""""""


def plotspearman(
        data,
        labels,
        cm="seismic",
        fons=20,
        figs=(
            20,
            17),
    tfs=25,
        ti="Spearman"):
    """Calculates Pearon matrix and returns it in the form of a plot.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    cm : TYPE, optional
        Color map as for matplolib. The default is "seismic".
    fons : TYPE, optional
        Plot font size. The default is 20.
    figs : TYPE, optional
        Plot size. The default is (20,17).
    tfs : TYPE, optional
        Title font size. The default is 25.
    ti : TYPE, optional
        Plot title/name. The default is "spearman".

    Returns
    -------
    Plot.
    """
    data = list(np.transpose(data))
    labels = list(labels)
    cm = str(cm)
    fonsize = float(fons)
    figsize = list(figs)
    titlefs = float(tfs)
    title = str(ti)
    n = len(data)

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
    fig = plt.figure(tight_layout=True, figsize=figsize)

    y = [i + 0.5 for i in range(n)]
    ticker = mpl.ticker.FixedLocator(y)
    formatter = mpl.ticker.FixedFormatter(labels)

    ax = fig.add_subplot(gs[0, 0])
    pcm = ax.pcolormesh(spear, cmap=cm, vmin=-1, vmax=1)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(title, fontsize=titlefs)
    ax.xaxis.set_major_locator(ticker)
    ax.yaxis.set_major_locator(ticker)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(rotation='90')
    plt.show()


def plotgrau(data, labels, cm="seismic", fons=20, figs=(25, 15),
             tfs=25, ti="Grau (Beta)", marker="s", marks=100):
    """Performs Grau correlation and plots it.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    cm : TYPE, optional
        DESCRIPTION. The default is "seismic".
    fons : TYPE, optional
        DESCRIPTION. The default is 20.
    figs : TYPE, optional
        DESCRIPTION. The default is (25,15).
    tfs : TYPE, optional
        DESCRIPTION. The default is 25.
    ti : TYPE, optional
        DESCRIPTION. The default is "Grau (Beta)".
    marker : TYPE, optional
        DESCRIPTION. The default is "s".
    marks : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    Plot.
    """
    data = list(np.transpose(data))
    labels = list(labels)
    cm = str(cm)  # color map as for matplolib
    fonsize = float(fons)  # plot fontsiez
    figsize = list(figs)  # plot size
    titlefs = float(tfs)  # title font size
    title = str(ti)  # plot name
    marker = str(marker)  # market style
    markersize = float(marks)  # marker size
    n = len(data)

    graus = grau(data)  # grau correlation (3d pearson)
    g1 = [graus[i][0] for i in range(len(graus))]  # first dimeniosn values
    g2 = [graus[i][1] for i in range(len(graus))]  # second dimension values
    g3 = [graus[i][2] for i in range(len(graus))]  # third dimension values
    mse = [graus[i][3] for i in range(len(graus))]  # mse's list
    g2_shift = list(g2)  # copy v2 to displace the d2 values for plotting
    t_c = []  # list of ticks per subplot
    xtick_labels = []  # list for labels
    c = 1  # number of different # in combs[i][0] (first column), starts with 1

    for i in range(len(graus) - 1):  # for all combinatios
        if graus[i][0] != graus[i + 1][0]:  # if it is different
            c += 1  # then it is a new one

    for i in range(c):  # for all the different first positions
        temp = 0  # temporal variable to count ticks
        for j in range(len(graus) - 1):  # check all the combinations
            if graus[j][0] == i:  # if it is the one we are looking for
                if graus[j][1] != graus[j +
                                        1][1]:  # if it changes, then new tick
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

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # (rows, columns)
    plt.rc('font', size=fonsize)
    fig = plt.figure(tight_layout=True, figsize=figsize)

    y = [i + 0.5 for i in range(n)]
    ticker = mpl.ticker.FixedLocator(y)
    formatter = mpl.ticker.FixedFormatter(labels)

    ax = fig.add_subplot(gs[0, 0])
    cm = plt.cm.get_cmap(cm)
    ax.set_title(title, fontsize=titlefs)
    sc = ax.scatter(g2_shift, g3, alpha=1, edgecolors='none',
                    c=mse, cmap=cm, s=markersize, marker=marker)
    plt.colorbar(sc)
    y_ticks = [i for i in range(int(min(g3)), int(max(g3)) + 1)]
    x_ticks = [i for i in range(int(max(g2_shift)) + 2)]
    ytick_labels = []
    for i in range(len(y_ticks)):
        ytick_labels.append(labels[y_ticks[i]])

    for i in range(len(xtick_labels) - 1):
        xtick_labels[i] = labels[xtick_labels[i]]
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
