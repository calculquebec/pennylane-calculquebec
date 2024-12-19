"""
Contains API utility functions and constants
"""

from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane.measurements import MeasurementProcess
import numpy as np
from base64 import b64encode

class ApiUtility:
    @staticmethod
    def convert_instruction(instruction : Operation) -> dict[str, any]:
        """converts a Pennylane operation to a dictionary that can be read by the Thunderhead API

        Args:
            instruction (Operation): a Pennylane operation (a gate)
            actual_qubits (list[int]): the mapping from the wires in pennylane to the wires in the physical machine

        Returns:
            dict[str, any]: a dictionary representation of the operation that can be read by the Thunderhead API
        """
        if instruction.name not in instructions:
            raise ValueError("This instruction is not supported")
        
        operation = {
            keys.QUBITS : [wire for wire in instruction.wires],
            keys.TYPE : instructions[instruction.name]
        }
        if instruction.name in parametered_ops: 
            value = instruction.parameters[0][0] if isinstance(instruction.parameters[0], np.ndarray) else instruction.parameters[0]
            operation[keys.PARAMETERS] = {"lambda" : value}
            
        return operation
    
    @staticmethod
    def convert_circuit(circuit : QuantumTape) -> dict[str, any]:
        """converts a pennylane quantum script to a dictionary that can be read by the Thunderhead API

        Args:
            tape (tape.QuantumScript): a pennylane quantum script (with informations about the number of wires, the operations and the measurements)

        Returns:
            dict[str, any]: a dictionary representation of the circuit that can be read by the API
        """

        circuit_dict = {
            keys.BIT_COUNT : 24,
            keys.OPERATIONS : [ApiUtility.convert_instruction(operation) for operation in circuit.operations],
            keys.QUBIT_COUNT : 24
        }
        wires = [wire for wire in circuit.wires]
        for measurement in circuit.measurements:
            for wire in measurement.wires:
                bit = wires.index(wire)
                circuit_dict[keys.OPERATIONS].append({
                    keys.QUBITS : [wire],
                    keys.BITS : [bit],
                    keys.TYPE : "readout"
                })
        return circuit_dict
    
    @staticmethod
    def basic_auth(username : str, password : str) -> str:
        """create a basic authentication token from a Thunderhead username and access token

        Args:
            username (str): your Thunderhead username
            password (str): your Thunderhead access token

        Returns:
            str: the basic authentification string that will authenticate you with the API
        """
        token = b64encode(f"{username}:{password}".encode('ascii')).decode("ascii")
        return f'Basic {token}'
    
    @staticmethod
    def headers(username : str, password : str, realm : str) -> dict[str, str]:
        """the Thunderhead API headers

        Args:
            username (str): your Thunderhead username
            password (str): your Thunderhead access token
            realm (str): your organization identifier with Thunderhead

        Returns:
            dict[str, any]: a dictionary representing the request headers
        """
        return {
            "Authorization" : ApiUtility.basic_auth(username, password),
            "Content-Type" : "application/json",
            "X-Realm" : realm
        }
    
    @staticmethod
    def job_body(circuit : dict[str, any], circuit_name : str, project_id : str, machine_name : str, shots) -> dict[str, any]:
        """the body for the job creation request

        Args:
            circuit (tape.QuantumScript): the script you want to convert
            name (str): the name of your job
            project_id (str): the id for the project for which this job will be run
            machine_name (str): the name of the machine on which this job will be run
            shots (int, optional): the number of shots (-1 will use the circuit's shot number)

        Returns:
            dict[str, any]: the body for the job creation request
        """
        body = {
            keys.NAME : circuit_name,
            keys.PROJECT_ID : project_id,
            keys.MACHINE_NAME : machine_name,
            keys.SHOT_COUNT : shots,
            keys.CIRCUIT : circuit,
        }
        return body

class routes:
    JOBS = "/jobs"
    PROJECTS = "/projects"
    MACHINES = "/machines"
    BENCHMARKING = "/benchmarking"
    MACHINE_NAME = "?machineName"
    
class keys:
    BIT_COUNT = "bitCount"
    QUBIT_COUNT = "qubitCount"
    OPERATIONS = "operations"
    CIRCUIT = "circuit"
    NAME = "name"
    MACHINE_NAME = "machineName"
    PROJECT_ID = "projectID"
    SHOT_COUNT = "shotCount"
    TYPE = "type"
    BITS = "bits"
    QUBITS = "qubits"
    PARAMETERS = "parameters"
    COUPLERS = "couplers"
    SINGLE_QUBIT_GATE_FIDELITY = "singleQubitGateFidelity"
    READOUT_STATE_0_FIDELITY = "readoutState0Fidelity"
    READOUT_STATE_1_FIDELITY = "readoutState1Fidelity"
    T1 = "t1"
    T2_RAMSEY = "t2Ramsey"
    CZ_GATE_FIDELITY = "czGateFidelity"
    RESULTS_PER_DEVICE = "resultsPerDevice"
    ITEMS = "items"
    ID = "id"

instructions : dict[str, str] = {
    "Identity" : "i",
    "PauliX" : "x",
    "X90" : "x_90",
    "XM90" : "x_minus_90",
    "PauliY" : "y",
    "Y90" : "y_90",
    "YM90" : "y_minus_90",
    "PauliZ" : "z",
    "Z90" : "z_90",
    "ZM90" : "z_minus_90",
    "T" : "t",
    "TDagger" : "t_dag",
    "CZ" : "cz",
    "PhaseShift" : "p",
    "RZ" : "rz"
}

parametered_ops : list[str] = [
    "RZ", 
    "PhaseShift"
]