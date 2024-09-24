from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane.measurements import MeasurementProcess

import requests
import json

class internal:
    @staticmethod
    def convert_instruction(instruction : Operation, actual_qubits : list[int]) -> dict[str, any]:
        """converts a Pennylane operation to a dictionary that can be read by the Thunderhead API

        Args:
            instruction (Operation): a Pennylane operation (a gate)
            actual_qubits (list[int]): the mapping from the wires in pennylane to the wires in the physical machine

        Returns:
            dict[str, any]: a dictionary representation of the operation that can be read by the Thunderhead API
        """
        operation = {
            internal.keys.qubits : [actual_qubits[i] for i, w in enumerate(instruction.wires)],
            internal.keys.type : instructions[instruction.name]
        }
        if instruction.name == "PhaseShift": operation[internal.keys.parameters] = {"lambda" : instruction.parameters[0]}
            
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
            internal.keys.bitCount : 24,
            internal.keys.operations : [internal.convert_instruction(op, circuit.wires) for op in circuit.operations if not isinstance(op, MeasurementProcess)],
            internal.keys.qubitCount : 24
        }
        for m in circuit.measurements:
            wires = m.wires if len(m.wires) > 0 else [_ for _ in range(len(circuit.wires))]
            for i, w in enumerate(wires):
                if i in [w2[internal.keys.qubits] for w2 in circuit_dict[internal.keys.operations]]:
                    continue
                circuit_dict[internal.keys.operations].append({
                    internal.keys.qubits : [circuit.wires[i]],
                    internal.keys.bits : [i],
                    internal.keys.type : "readout"
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
        from base64 import b64encode
        token = b64encode(f"{username}:{password}".encode('ascii')).decode("ascii")
        return f'Basic {token}'
    
    @staticmethod
    def headers(username : str, password : str, realm : str) -> dict[str, any]:
        """the Thunderhead API headers

        Args:
            username (str): your Thunderhead username
            password (str): your Thunderhead access token
            realm (str): your organization identifier with Thunderhead

        Returns:
            dict[str, any]: a dictionary representing the request headers
        """
        return {
            "Authorization" : internal.basic_auth(username, password),
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
            internal.keys.name : circuit_name,
            internal.keys.projectID : project_id,
            internal.keys.machineName : machine_name,
            internal.keys.shotCount : shots,
            internal.keys.circuit : circuit,
        }
        return body

    class routes:
        jobs = "/jobs"
        projects = "/projects"
        
    class keys:
        bitCount = "bitCount"
        qubitCount = "qubitCount"
        operations = "operations"
        circuit = "circuit"
        name = "name"
        machineName = "machineName"
        projectID = "projectID"
        shotCount = "shotCount"
        type = "type"
        bits = "bits"
        qubits = "qubits"
        parameters = "parameters"


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
    "Hadamard" : "h",
    "CNOT" : "cnot"
}
    

class ApiAdapter:
    def __init__(self, host = "", user = "", access_token = "", realm = ""):
        self.host = host
        self.headers = internal.headers(user, access_token, realm)
        
    def create_job(self, circuit : dict[str, any], 
                   circuit_name : str, 
                   project_id : str, 
                   machine_name : str, 
                   shot_count : int) -> requests.Response:
        body = internal.job_body(circuit, circuit_name, project_id, machine_name, shot_count)
        return requests.post(self.host + internal.routes.jobs, data=json.dumps(body), headers=self.headers)

    def list_jobs(self) -> requests.Response:
        return requests.get(self.host + internal.routes.jobs, headers=self.headers)

    def job_by_id(self, id : str) -> requests.Response:
        return requests.get(self.host + internal.routes.jobs + f"/{id}", headers=self.headers)
