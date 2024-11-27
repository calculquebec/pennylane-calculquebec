from dotenv import dotenv_values
from pennylane_snowflurry.API.api_adapter import ApiAdapter
from pennylane_snowflurry.utility.api_utility import keys

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

def build_benchmark(q1Acceptance, q2Acceptance):
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
