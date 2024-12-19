"""
a utility module that contains noise related functions
"""

import numpy as np

def depolarizing_noise(fidelity):
    """
    Calculates the probability of depolarizing noise
    given the fidelity of the gate.
    """
    return 1 - fidelity

def amplitude_damping(t, t1):
    """Compute amplitude damping parameter gamma."""
    return 1 - np.exp(-t / t1) if t1 > 0 else None

def phase_damping(t, t2):
    """Compute phase damping parameter lambda."""
    return 1 - np.exp(-t / t2) if t2 > 0 else None

class TypicalBenchmark:
    """
    typical errors represented as constants
    """
    qubit = 0.99
    cz = 0.95
    readout0 = 0.95
    readout1 = 0.85
    t1 = 9.6E-6
    t2Ramsey = 2.3E-6
    
def readout_error(state0, state1):
    """
    a readout error matrix
    
    Args:
        state0 (float) : state 0 readout error
        state1 (float) : state 1 readout error
    
    Returns:
        np.array : a 2 x 2 array representing provided readout errors
    """
    return np.array([
            [state0, 1 - state1],
            [1 - state0, state1]
        ])