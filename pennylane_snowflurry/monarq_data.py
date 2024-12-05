"""
Contains MonarQ's connectivity + functions for retreiving broken qubits and couplers and cz / state 1 fidelities
"""

from pennylane_snowflurry.API.adapter import ApiAdapter
from pennylane_snowflurry.utility.api import keys
from pennylane_snowflurry.utility.noise import depolarizing_noise, phase_damping, amplitude_damping
import numpy as np

"""
#       00
#       |
#    08-04-01
#    |  |  | 
# 16-12-09-05-02
# |  |  |  |  |
# 20-17-13-10-06-03
#    |  |  |  |  |
#    21-18-14-11-07
#       |  |  |
#       22-19-15
#          |
#          23
"""
connectivity = {
  keys.qubits : [ 0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23 ],
  
  keys.couplers : {
      "0": [0, 4],
      "1": [4, 1],
      "2": [1, 5],
      "3": [5, 2],
      "4": [2, 6],
      "5": [6, 3],
      "6": [3, 7],
      "7": [8, 4],
      "8": [4, 9],
      "9": [9, 5],
      "10": [5, 10],
      "11": [10, 6],
      "12": [6, 11],
      "13": [11, 7],
      "14": [8, 12],
      "15": [12, 9],
      "16": [9, 13],
      "17": [13, 10],
      "18": [10, 14],
      "19": [14, 11],
      "20": [11, 15],
      "21": [16, 12],
      "22": [12, 17],
      "23": [17, 13],
      "24": [13, 18],
      "25": [18, 14],
      "26": [14, 19],
      "27": [19, 15],
      "28": [16, 20],
      "29": [20, 17],
      "30": [17, 21],
      "31": [21, 18],
      "32": [18, 22],
      "33": [22, 19],
      "34": [19, 23]
  }
}

class cache:
    _readout1_cz_fidelities : dict = None
    _relaxation : list = None
    _decoherence : list = None
    _qubit_noise : list = None
    _coupler_noise : list = None
    _readout_noise : list = None

def get_broken_qubits_and_couplers(q1Acceptance, q2Acceptance):
    """
    creates a dictionary that contains unreliable qubits and couplers
    """
    val = (q1Acceptance, q2Acceptance)

    # call to api to get qubit and couplers benchmark
    qubits_and_couplers = ApiAdapter.get_qubits_and_couplers()

    broken_qubits_and_couplers = { keys.qubits : [], keys.couplers : [] }

    for coupler_id in qubits_and_couplers[keys.couplers]:
        benchmark_coupler = qubits_and_couplers[keys.couplers][coupler_id]
        conn_coupler = connectivity[keys.couplers][coupler_id]

        if benchmark_coupler[keys.czGateFidelity] >= val[1]:
            continue

        broken_qubits_and_couplers[keys.couplers].append(conn_coupler)

    for qubit_id in qubits_and_couplers[keys.qubits]:
        benchmark_qubit = qubits_and_couplers[keys.qubits][qubit_id]

        if benchmark_qubit[keys.readoutState1Fidelity] >= val[0]:
            continue

        broken_qubits_and_couplers[keys.qubits].append(int(qubit_id))
    return broken_qubits_and_couplers

def get_readout1_and_cz_fidelities():
    """get state 1 fidelities and cz fidelities
    """
    if cache._readout1_cz_fidelities is None:
        cache._readout1_cz_fidelities = {keys.readoutState1Fidelity:{}, keys.czGateFidelity:{}}
        benchmark = ApiAdapter.get_benchmark()[keys.resultsPerDevice]
    
        # build state 1 fidelity
        for key in benchmark[keys.qubits]:
            cache._readout1_cz_fidelities[keys.readoutState1Fidelity][key] = benchmark[keys.qubits][key][keys.readoutState1Fidelity]
        
        # build cz fidelity
        for key in benchmark[keys.couplers]:
            link = connectivity[keys.couplers][key]
            cache._readout1_cz_fidelities[keys.czGateFidelity][(link[0], link[1])] = benchmark[keys.couplers][key][keys.czGateFidelity]
        
    return cache._readout1_cz_fidelities

def get_qubit_and_coupler_noise():
    
    if cache._qubit_noise is None or cache._coupler_noise is None or ApiAdapter.is_last_update_expired():
        benchmark = ApiAdapter.get_qubits_and_couplers()
    
        single_qubit_gate_fidelity = {} 
        cz_gate_fidelity = {}
        num_qubits = len(benchmark[keys.qubits])
        num_couplers = len(benchmark[keys.couplers])

        for i in range(num_qubits):
            single_qubit_gate_fidelity[i] = benchmark[keys.qubits][str(i)][keys.singleQubitGateFidelity]
        single_qubit_gate_fidelity = list(single_qubit_gate_fidelity.values())   

        for i in range(num_couplers):
            cz_gate_fidelity[i] = benchmark[keys.couplers][str(i)][keys.czGateFidelity]
        cz_gate_fidelity = list(cz_gate_fidelity.values())   

        cache._qubit_noise = [
            depolarizing_noise(fidelity) if fidelity > 0 else None 
            for fidelity in single_qubit_gate_fidelity
        ]

        coupler_noise_array = [
            depolarizing_noise(fidelity) if fidelity > 0 else None 
            for fidelity in cz_gate_fidelity
        ]
        cache._coupler_noise = { }
        for i, noise in enumerate(coupler_noise_array):
            link = connectivity[keys.couplers][str(i)]
            cache._coupler_noise[(link[0], link[1])] = noise
            
            
    return cache._qubit_noise, cache._coupler_noise

def get_amplitude_and_phase_damping():
    
    if cache._relaxation is None or cache._decoherence is None or ApiAdapter.is_last_update_expired():
        benchmark = ApiAdapter.get_qubits_and_couplers()
        time_step = 1e-6 # microsecond
        num_qubits = len(benchmark[keys.qubits])

        t1_values = {}
        for i in range(num_qubits):
            t1_values[i] = benchmark[keys.qubits][str(i)][keys.t1]
        t1_values = list(t1_values.values())  

        t2_values = {}
        for i in range(num_qubits):
            t2_values[i] = benchmark[keys.qubits][str(i)][keys.t2Ramsey]
        t2_values = list(t2_values.values())  

        cache._relaxation = [
            amplitude_damping(time_step, t1) for t1 in t1_values
        ]

        cache._decoherence = [
            phase_damping(time_step, t2) for t2 in t2_values
        ]
    return cache._relaxation, cache._decoherence


def get_readout_noise_matrices():
    
    if cache._readout_noise is None or ApiAdapter.is_last_update_expired():
        benchmark = ApiAdapter.get_qubits_and_couplers()
        num_qubits = len(benchmark[keys.qubits])

        readout_state_0_fidelity = []
        readout_state_1_fidelity = []
        
        for i in range(num_qubits):
            readout_state_0_fidelity.append(benchmark[keys.qubits][str(i)][keys.readoutState0Fidelity])
            readout_state_1_fidelity.append(benchmark[keys.qubits][str(i)][keys.readoutState1Fidelity])

        cache._readout_noise = []

        for f0, f1 in zip(readout_state_0_fidelity, readout_state_1_fidelity):
            R = np.array([
                [f0, 1 - f1],
                [1 - f0, f1]
            ])
            cache._readout_noise.append(R)
    return cache._readout_noise