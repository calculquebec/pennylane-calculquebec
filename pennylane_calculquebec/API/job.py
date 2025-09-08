"""
Contains a wrapper around the job creation and executing process for MonarQ
"""

from pennylane.tape import QuantumTape
import json
import time
from pennylane_calculquebec.API.adapter import ApiAdapter
from pennylane_calculquebec.utility.api import ApiUtility, JobStatus
from typing import Callable


class JobException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message


class Job:
    """A wrapper around Thunderhead's jobs operations.
    - converts your circuit to an http request
    - posts a job on monarq
    - periodically checks if the job is done
    - returns results when it's done

    Args:
        circuit (QuantumTape) : the circuit you want to execute
    """

    started: Callable[[int], None]
    status_changed: Callable[[int, str], None]
    completed: Callable[[int], None]

    def __init__(
        self,
        circuit: QuantumTape,
    ):
        self.started = None
        self.status_changed = None
        self.completed = None
        self.circuit_dict = ApiUtility.convert_circuit(circuit)
        self.shots = circuit.shots.total_shots

    def run(self, max_tries: int = 2**15) -> dict:
        """
        converts a quantum tape into a dictionary, readable by thunderhead
        creates a job on thunderhead
        fetches the result until the job is successfull, and returns the result

        Args:
            max_tries (int) : the number of tries before dropping a circuit. Defaults to 2 ^ 15

        Args:
            max_tries (int) : the number of tries before dropping a circuit. Defaults to 2 ^ 15
        """

        response = None
        try:
            response = ApiAdapter.post_job(
                self.circuit_dict,
                self.shots,
            )
        except:
            raise
        if response.status_code == 200:

            current_status = ""
            job_id = json.loads(response.text)["job"]["id"]
            if self.started is not None:
                self.started(job_id)
            for i in range(max_tries):
                time.sleep(0.2)
                response = ApiAdapter.job_by_id(job_id)

                if response.status_code != 200:
                    self.raise_api_error(response)

                content = json.loads(response.text)
                status = content["job"]["status"]["type"]
                if current_status != status:

                    current_status = status
                    if self.status_changed is not None:
                        self.status_changed(job_id, status)

                if status != JobStatus.SUCCEEDED.value:
                    continue
                if self.completed is not None:
                    self.completed(job_id)

                return content["result"]["histogram"]
            raise JobException(
                "Couldn't finish job. Stuck on status : " + str(current_status)
            )
        else:
            self.raise_api_error(response)

    def raise_api_error(self, response):
        """
        this raises an error by parsing the json body of the response, and using the response text as message

        Args:
            response (Response) : the erroneous response

        Raises:
            - JobException
        """
        message = response

        # try to fetch the text from the response
        if hasattr(message, "text"):
            message = message.text

        # try to deserialize the text (it might not be deserializable)
        try:
            message = json.loads(message)
            if "error" in message:
                message = message["error"]
        except Exception:
            pass

        raise JobException(message)
