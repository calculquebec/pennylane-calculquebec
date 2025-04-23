"""
Contains MonarQ's connectivity + benchmarking functionalities
"""

from pennylane_calculquebec.API.adapter import ApiAdapter
from pennylane_calculquebec.utility.api import keys
from pennylane_calculquebec.utility.noise import depolarizing_noise, phase_damping, amplitude_damping
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

class Cache:
    READOUT1_CZ = "readout1_cz"
    RELAXATION = "relaxation"
    DECOHERENCE = "decoherence"
    QUBIT_NOISE = "qubit_noise"
    COUPLER_NOISE = "coupler_noise"
    READOUT_NOISE = "readout_noise"
    CONNECTIVITY = "connectivity"
    OFFLINE_CONNECTIVITY = "offline_connectivity"

cache = {
    "yamaska" : {
        Cache.OFFLINE_CONNECTIVITY : {
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
    },
    "yukon" : {
        Cache.OFFLINE_CONNECTIVITY : {
            "0" : [0, 1],
            "1" : [1, 2],
            "2" : [2, 3],
            "3" : [3, 4],
            "4" : [4, 5],
        }
    }
}
def is_cache_out_of_date(machine_name : str, cache_element : str):
    return ApiAdapter.is_last_update_expired() \
        or machine_name not in cache \
        or cache_element not in cache[machine_name]

def monarq_native_gates():
    """the names of all gates in the MonarQ native gate set

    Returns:
        list[str]: the gates' names
    """
    return [
            "T", "TDagger",
            "PauliX", "PauliY", "PauliZ", 
            "X90", "Y90", "Z90",
            "XM90", "YM90", "ZM90",
            "PhaseShift", "CZ", "RZ", "Identity"
        ]

def get_connectivity(machine_name, use_benchmark = True):
    """get the connectivity for a given machine

    Args:
        machine_name (str): the name of the machine. usually yukon or yamaska.
        use_benchmark (bool, optional): should we use api calls to get the connectivity. Defaults to True.

    Returns:
        dict: the couplers in a given machine
    """
    if not use_benchmark:
        return cache[machine_name][Cache.OFFLINE_CONNECTIVITY]
    
    if is_cache_out_of_date(machine_name, Cache.CONNECTIVITY):
        cache[machine_name][Cache.CONNECTIVITY] = ApiAdapter.get_connectivity_for_machine(machine_name)
    return cache[machine_name][Cache.CONNECTIVITY]
    
def get_broken_qubits_and_couplers(q1Acceptance, q2Acceptance, machine_name):
    """
    creates a dictionary that contains unreliable qubits and couplers

    Args:
        q1Acceptance (float) : what fidelity should be considered broken for a qubit?
        q2Acceptance (float) : what fidelity should be considered broken for a coupler?

    Returns:
        dict : which qubits and couplers can be used, given arbitrary acceptance values
    """
    val = (q1Acceptance, q2Acceptance)
    
    # call to api to get qubit and couplers benchmark
    qubits_and_couplers = ApiAdapter.get_qubits_and_couplers(machine_name)

    broken_qubits_and_couplers = { keys.QUBITS : [], keys.COUPLERS : [] }

    for coupler_id in qubits_and_couplers[keys.COUPLERS]:
        benchmark_coupler = qubits_and_couplers[keys.COUPLERS][coupler_id]
        conn_coupler = get_connectivity(machine_name)[coupler_id]

        if benchmark_coupler[keys.CZ_GATE_FIDELITY] >= val[1]:
            continue

        broken_qubits_and_couplers[keys.COUPLERS].append(conn_coupler)

    for qubit_id in qubits_and_couplers[keys.QUBITS]:
        benchmark_qubit = qubits_and_couplers[keys.QUBITS][qubit_id]

        if benchmark_qubit[keys.READOUT_STATE_1_FIDELITY] >= val[0]:
            continue

        broken_qubits_and_couplers[keys.QUBITS].append(int(qubit_id))
    return broken_qubits_and_couplers

def get_readout1_and_cz_fidelities(machine_name):
    """
    get state 1 fidelities and cz fidelities\n
    example : {"readoutState1Fidelity" : {"0" : 1}, "czGateFidelity" : {(0, 1) : 1}}

    Args:
        machine_name (str) : the name of the machine. usually yukon or yamaska

    Returns:
        dict[str, dict[str, float] | dict[tuple[int], float]] : fidelity values for readout1 and couplers
    """
    if is_cache_out_of_date(machine_name, Cache.READOUT1_CZ):
        cache[machine_name][Cache.READOUT1_CZ] = {keys.READOUT_STATE_1_FIDELITY:{}, keys.CZ_GATE_FIDELITY:{}}
        benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
    
        # build state 1 fidelity
        for key in benchmark[keys.QUBITS]:
            cache[machine_name][Cache.READOUT1_CZ][keys.READOUT_STATE_1_FIDELITY][key] = benchmark[keys.QUBITS][key][keys.READOUT_STATE_1_FIDELITY]
        
        # build cz fidelity
        for key in benchmark[keys.COUPLERS]:
            link = get_connectivity(machine_name)[key]
            cache[machine_name][Cache.READOUT1_CZ][keys.CZ_GATE_FIDELITY][(link[0], link[1])] = benchmark[keys.COUPLERS][key][keys.CZ_GATE_FIDELITY]
        
    return cache[machine_name][Cache.READOUT1_CZ]

def get_coupler_noise(machine_name) -> dict:
    """
    build cz gate error array
    
    Args:
        machine_name (str) : the name of the machine. Usually yukon or yamaska

    Returns :
        dict[Tuple[int, int], float] : a dictionary of links and values representing cz gate errors
    """
    if is_cache_out_of_date(machine_name, Cache.COUPLER_NOISE):
        benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
    
        cz_gate_fidelity = {}
        num_couplers = len(benchmark[keys.COUPLERS])

        for i in range(num_couplers):
            cz_gate_fidelity[i] = benchmark[keys.COUPLERS][str(i)][keys.CZ_GATE_FIDELITY]
        cz_gate_fidelity = list(cz_gate_fidelity.values())   

        coupler_noise_array = [
            depolarizing_noise(fidelity) if fidelity > 0 else None 
            for fidelity in cz_gate_fidelity
        ]
        cache[machine_name][Cache.COUPLER_NOISE] = { }
        for i, noise in enumerate(coupler_noise_array):
            link = get_connectivity(machine_name)[str(i)]
            cache[machine_name][Cache.COUPLER_NOISE][(link[0], link[1])] = noise
            
            
    return cache[machine_name][Cache.COUPLER_NOISE]

def get_qubit_noise(machine_name):
    """
    build single qubit gate error array
    
    Args:
        machine_name (str) : the name of the machine. Usually yukon or yamaska
        
    Returns :
        list[float] : a list of values representing single qubit gate errors
    """
    if is_cache_out_of_date(machine_name, Cache.QUBIT_NOISE):
        benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
    
        single_qubit_gate_fidelity = {} 

        num_qubits = len(benchmark[keys.QUBITS])

        for i in range(num_qubits):
            single_qubit_gate_fidelity[i] = benchmark[keys.QUBITS][str(i)][keys.SINGLE_QUBIT_GATE_FIDELITY]
        single_qubit_gate_fidelity = list(single_qubit_gate_fidelity.values())   

        cache[machine_name][Cache.QUBIT_NOISE] = [
            depolarizing_noise(fidelity) if fidelity > 0 else None 
            for fidelity in single_qubit_gate_fidelity
        ]
            
    return cache[machine_name][Cache.QUBIT_NOISE]

def get_phase_damping(machine_name):
    """
    builds decoherence error arrays using t2 time

    Args:
        machine_name (str) : the name of the machine. Usually yukon or yamaska
        
    Returns:
        list[float] : decoherence values for each qubit
    """
    if is_cache_out_of_date(machine_name, Cache.DECOHERENCE):
        benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
        time_step = 1e-6 # microsecond
        num_qubits = len(benchmark[keys.QUBITS])

        t2_values = {}
        for i in range(num_qubits):
            t2_values[i] = benchmark[keys.QUBITS][str(i)][keys.T2_RAMSEY]
        t2_values = list(t2_values.values())  

        cache[machine_name][Cache.DECOHERENCE] = [
            phase_damping(time_step, t2) for t2 in t2_values
        ]
    return cache[machine_name][Cache.DECOHERENCE]

def get_amplitude_damping(machine_name):
    """
    builds relaxation error arrays using t1 time

    Args:
        machine_name (str) : the name of the machine. Usually yukon or yamaska
        
    Returns:
        list[float] : relaxation values for each qubit
    """
    if is_cache_out_of_date(machine_name, Cache.RELAXATION):
        benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
        time_step = 1e-6 # microsecond
        num_qubits = len(benchmark[keys.QUBITS])

        t1_values = {}
        for i in range(num_qubits):
            t1_values[i] = benchmark[keys.QUBITS][str(i)][keys.T1]
        t1_values = list(t1_values.values())  

        cache[machine_name][Cache.RELAXATION] = [
            amplitude_damping(time_step, t1) for t1 in t1_values
        ]

    return cache[machine_name][Cache.RELAXATION]

def get_readout_noise_matrices(machine_name):
    """
    constructs an array of readout noise matrices
    
    Args:
        machine_name (str) : the name of the machine. Usually yukon or yamaska
    
    Returns:
        np.ndarray : an array of 2x2 matrices built from state 0 / 1 fidelities
    """
    if is_cache_out_of_date(machine_name, Cache.READOUT_NOISE):
        benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
        num_qubits = len(benchmark[keys.QUBITS])

        readout_state_0_fidelity = []
        readout_state_1_fidelity = []
        
        for i in range(num_qubits):
            readout_state_0_fidelity.append(benchmark[keys.QUBITS][str(i)][keys.READOUT_STATE_0_FIDELITY])
            readout_state_1_fidelity.append(benchmark[keys.QUBITS][str(i)][keys.READOUT_STATE_1_FIDELITY])

        cache[machine_name][Cache.READOUT_NOISE] = []

        for f0, f1 in zip(readout_state_0_fidelity, readout_state_1_fidelity):
            R = np.array([
                [f0, 1 - f1],
                [1 - f0, f1]
            ])
            cache[machine_name][Cache.READOUT_NOISE].append(R)
    return cache[machine_name][Cache.READOUT_NOISE]
