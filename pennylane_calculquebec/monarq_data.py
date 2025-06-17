"""
Contains MonarQ's connectivity + benchmarking functionalities
"""

from pennylane_calculquebec.API.adapter import ApiAdapter
from pennylane_calculquebec.utility.api import keys
from pennylane_calculquebec.utility.noise import (
    depolarizing_noise,
    phase_damping,
    amplitude_damping,
)
import numpy as np
from pennylane_calculquebec.logger import logger

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
    "yamaska": {
        Cache.OFFLINE_CONNECTIVITY: {
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
            "34": [19, 23],
        }
    },
    "yukon": {
        Cache.OFFLINE_CONNECTIVITY: {
            "0": [0, 1],
            "1": [1, 2],
            "2": [2, 3],
            "3": [3, 4],
            "4": [4, 5],
        }
    },
}


def is_cache_out_of_date(machine_name: str, cache_element: str):
    try:
        return (
            ApiAdapter.is_last_update_expired()
            or machine_name not in cache
            or cache_element not in cache[machine_name]
        )
    except Exception as e:
        logger.error(
            "Error %s in is_cache_out_of_date located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return True


def monarq_native_gates():
    try:
        return [
            "T",
            "TDagger",
            "PauliX",
            "PauliY",
            "PauliZ",
            "X90",
            "Y90",
            "Z90",
            "XM90",
            "YM90",
            "ZM90",
            "PhaseShift",
            "CZ",
            "RZ",
            "Identity",
        ]
    except Exception as e:
        logger.error(
            "Error %s in monarq_native_gates located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return []


def get_connectivity(machine_name, use_benchmark=True):
    try:
        if not use_benchmark:
            return cache[machine_name][Cache.OFFLINE_CONNECTIVITY]

        if is_cache_out_of_date(machine_name, Cache.CONNECTIVITY):
            cache[machine_name][Cache.CONNECTIVITY] = (
                ApiAdapter.get_connectivity_for_machine(machine_name)
            )
        return cache[machine_name][Cache.CONNECTIVITY]
    except Exception as e:
        logger.error(
            "Error %s in get_connectivity located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return {}


def get_broken_qubits_and_couplers(q1Acceptance, q2Acceptance, machine_name):
    try:
        val = (q1Acceptance, q2Acceptance)
        qubits_and_couplers = ApiAdapter.get_qubits_and_couplers(machine_name)
        broken_qubits_and_couplers = {keys.QUBITS: [], keys.COUPLERS: []}

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
    except Exception as e:
        logger.error(
            "Error %s in get_broken_qubits_and_couplers located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return {keys.QUBITS: [], keys.COUPLERS: []}


def get_readout1_and_cz_fidelities(machine_name):
    try:
        if is_cache_out_of_date(machine_name, Cache.READOUT1_CZ):
            cache[machine_name][Cache.READOUT1_CZ] = {
                keys.READOUT_STATE_1_FIDELITY: {},
                keys.CZ_GATE_FIDELITY: {},
            }
            benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
            for key in benchmark[keys.QUBITS]:
                cache[machine_name][Cache.READOUT1_CZ][keys.READOUT_STATE_1_FIDELITY][
                    key
                ] = benchmark[keys.QUBITS][key][keys.READOUT_STATE_1_FIDELITY]
            for key in benchmark[keys.COUPLERS]:
                link = get_connectivity(machine_name)[key]
                cache[machine_name][Cache.READOUT1_CZ][keys.CZ_GATE_FIDELITY][
                    (link[0], link[1])
                ] = benchmark[keys.COUPLERS][key][keys.CZ_GATE_FIDELITY]
        return cache[machine_name][Cache.READOUT1_CZ]
    except Exception as e:
        logger.error(
            "Error %s in get_readout1_and_cz_fidelities located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return {}


def get_coupler_noise(machine_name) -> dict:
    try:
        if is_cache_out_of_date(machine_name, Cache.COUPLER_NOISE):
            benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
            cz_gate_fidelity = {}
            num_couplers = len(benchmark[keys.COUPLERS])
            for i in range(num_couplers):
                cz_gate_fidelity[i] = benchmark[keys.COUPLERS][str(i)][
                    keys.CZ_GATE_FIDELITY
                ]
            cz_gate_fidelity = list(cz_gate_fidelity.values())
            coupler_noise_array = [
                depolarizing_noise(fidelity) if fidelity > 0 else None
                for fidelity in cz_gate_fidelity
            ]
            cache[machine_name][Cache.COUPLER_NOISE] = {}
            for i, noise in enumerate(coupler_noise_array):
                link = get_connectivity(machine_name)[str(i)]
                cache[machine_name][Cache.COUPLER_NOISE][(link[0], link[1])] = noise
        return cache[machine_name][Cache.COUPLER_NOISE]
    except Exception as e:
        logger.error(
            "Error %s in get_coupler_noise located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return {}


def get_qubit_noise(machine_name):
    try:
        if is_cache_out_of_date(machine_name, Cache.QUBIT_NOISE):
            benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
            single_qubit_gate_fidelity = {}
            num_qubits = len(benchmark[keys.QUBITS])
            for i in range(num_qubits):
                single_qubit_gate_fidelity[i] = benchmark[keys.QUBITS][str(i)][
                    keys.SINGLE_QUBIT_GATE_FIDELITY
                ]
            single_qubit_gate_fidelity = list(single_qubit_gate_fidelity.values())
            cache[machine_name][Cache.QUBIT_NOISE] = [
                depolarizing_noise(fidelity) if fidelity > 0 else None
                for fidelity in single_qubit_gate_fidelity
            ]
        return cache[machine_name][Cache.QUBIT_NOISE]
    except Exception as e:
        logger.error(
            "Error %s in get_qubit_noise located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return []


def get_phase_damping(machine_name):
    try:
        if is_cache_out_of_date(machine_name, Cache.DECOHERENCE):
            benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
            time_step = 1e-6  # microsecond
            num_qubits = len(benchmark[keys.QUBITS])
            t2_values = {}
            for i in range(num_qubits):
                t2_values[i] = benchmark[keys.QUBITS][str(i)][keys.T2_RAMSEY]
            t2_values = list(t2_values.values())
            cache[machine_name][Cache.DECOHERENCE] = [
                phase_damping(time_step, t2) for t2 in t2_values
            ]
        return cache[machine_name][Cache.DECOHERENCE]
    except Exception as e:
        logger.error(
            "Error %s in get_phase_damping located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return []


def get_amplitude_damping(machine_name):
    try:
        if is_cache_out_of_date(machine_name, Cache.RELAXATION):
            benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
            time_step = 1e-6  # microsecond
            num_qubits = len(benchmark[keys.QUBITS])
            t1_values = {}
            for i in range(num_qubits):
                t1_values[i] = benchmark[keys.QUBITS][str(i)][keys.T1]
            t1_values = list(t1_values.values())
            cache[machine_name][Cache.RELAXATION] = [
                amplitude_damping(time_step, t1) for t1 in t1_values
            ]
        return cache[machine_name][Cache.RELAXATION]
    except Exception as e:
        logger.error(
            "Error %s in get_amplitude_damping located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return []


def get_readout_noise_matrices(machine_name):
    try:
        if is_cache_out_of_date(machine_name, Cache.READOUT_NOISE):
            benchmark = ApiAdapter.get_qubits_and_couplers(machine_name)
            num_qubits = len(benchmark[keys.QUBITS])
            readout_state_0_fidelity = []
            readout_state_1_fidelity = []
            for i in range(num_qubits):
                readout_state_0_fidelity.append(
                    benchmark[keys.QUBITS][str(i)][keys.READOUT_STATE_0_FIDELITY]
                )
                readout_state_1_fidelity.append(
                    benchmark[keys.QUBITS][str(i)][keys.READOUT_STATE_1_FIDELITY]
                )
            cache[machine_name][Cache.READOUT_NOISE] = []
            for f0, f1 in zip(readout_state_0_fidelity, readout_state_1_fidelity):
                R = np.array([[f0, 1 - f1], [1 - f0, f1]])
                cache[machine_name][Cache.READOUT_NOISE].append(R)
        return cache[machine_name][Cache.READOUT_NOISE]
    except Exception as e:
        logger.error(
            "Error %s in get_readout_noise_matrices located in monarq_data: %s",
            type(e).__name__,
            e,
        )
        return []
