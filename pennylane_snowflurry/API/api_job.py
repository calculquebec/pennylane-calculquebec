from pennylane.tape import QuantumTape
import json
import time
from pennylane_snowflurry.API.api_adapter import ApiAdapter
from pennylane_snowflurry.utility.api_utility import ApiUtility

class JobException(Exception):
    def __init__(self, message : str):
        self.message = message
    
    def __str__(self): self.message
    
class Job:
    """A wrapper around Thunderhead's jobs operations. 
    - converts your circuit to an http request
    - posts a job on monarq
    - periodically checks if the job is done
    - returns results when it's done

    Args : 
        - circuit (QuantumTape) : the circuit you want to execute
        - circuit_name (str) : the name of the circuit
    """
    
    def __init__(self, circuit : QuantumTape, circuit_name = "default"):
        self.circuit_dict = ApiUtility.convert_circuit(circuit)
        self.circuit_name = circuit_name
        self.shots = circuit.shots.total_shots

    def run(self, max_tries : int = -1) -> dict:
        """
        converts a quantum tape into a dictionary, readable by thunderhead
        creates a job on thunderhead
        fetches the result until the job is successfull, and returns the result
        """

        if max_tries == -1: max_tries = 2 ** 15
        response = None
        try:
            response = ApiAdapter.create_job(self.circuit_dict, self.circuit_name, self.shots)
        except:
            raise
        if(response.status_code == 200):
            current_status = ""
            job_id = json.loads(response.text)["job"]["id"]
            for i in range(max_tries):
                time.sleep(0.2)
                response = ApiAdapter.job_by_id(job_id)

                if response.status_code != 200: 
                    continue

                content = json.loads(response.text)
                status = content["job"]["status"]["type"]
                if(current_status != status):
                    current_status = status

                if(status != "SUCCEEDED"): 
                    continue

                return content["result"]["histogram"]
            raise JobException("Couldn't finish job. Stuck on status : " + str(current_status))
        else:
            raise JobException(response.text)