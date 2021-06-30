"""
This main module contains all the functions available in spectrapepper. Please use the search function to look up
specific functionalities with key words.
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


def load(file, fromline=0, transpose=False):
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
            s_row = np.array(s_row, dtype=float)
            new_data.append(s_row)
        i += 1
    raw_data.close()

    if transpose:
        new_data = np.transpose(new_data)

    return new_data


def loadline(file, line=0, tp='float', split=False):
    """
    Random access to file. Loads a specific line in a file. Useful when
    managing large data files in processes where time is important. It can
    load numbers as floats.

    :type file: str
    :param file: Url od the data file

    :type line: int
    :param line: Line number. Counts from 0.

    :type tp: str
    :param tp: Type of data. If its numeric then 'float', if text then 'string'.
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

    if tp == 'float':
        info = info.replace("NaN", "-1")
        info = info.replace("nan", "-1")
        info = info.replace("--", "-1")
        info = str.split(info)
        info = np.array(info, dtype=float)

    if tp == 'string':
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
    data = loadline(my_file, 1, tp='string', split=True)
    r = data[2]
    if r != 'second':
        default = False

    return default
