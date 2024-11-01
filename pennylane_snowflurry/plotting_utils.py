import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from typing import Callable

def graph(func : Callable, array_dict : dict[str, list], x_axis : list, title="", xlabel="", ylabel=""):
    for key in array_dict:
        func(x_axis, array_dict[key], label = key)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def fit(x_axis : list, array : list, label : str):
    def fit_func(x, a, b): return a*(x**b)
    
    params, _ = curve_fit(fit_func, x_axis, array)        
    a, b = params
    
    plt.scatter(x_axis, array, label=label)
    plt.plot(x_axis, fit_func(x_axis, a, b), label=f"fitted {label}: $y={a:.3f} x^{{{b:.2f}}}$")
