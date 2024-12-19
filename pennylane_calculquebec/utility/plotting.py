"""
Contains utility functions for plotting\n
this is not used in any feature.\n
it is mostly used for developpers to simplify plotting
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from typing import Callable

def  graph(func : Callable, array_dict : dict[str, list], x_axis : list, title="", xlabel="", ylabel=""):
    """a general function for drawing graphs. Mainly used to compare multiple values on one graph

    Args:
        func (Callable): the plt method (bar, hist, ...)
        array_dict (dict[str, list]): labels and list of values
        x_axis (list): list of discrete values that the different labels will take
        title (str, optional): title of the graph. Defaults to "".
        xlabel (str, optional): the name of the x axis. Defaults to "".
        ylabel (str, optional): the name of the y axis. Defaults to "".
    """
    for key in array_dict:
        func(x_axis, array_dict[key], label = key)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def fit(x_axis : list, array : list, label : str):
    """generates a fit graph (a scatter plot with a curve that follows the general tendency of the points)

    Args:
        x_axis (list): the independent variable values
        array (list): the dependent variable values
        label (str): the name of the data set
    """
    def fit_func(x, a, b): return a*(x**b)
    
    params, _ = curve_fit(fit_func, x_axis, array)        
    a, b = params
    
    plt.scatter(x_axis, array, label=label)
    plt.plot(x_axis, fit_func(x_axis, a, b), label=f"fitted {label}: $y={a:.3f} x^{{{b:.2f}}}$")
