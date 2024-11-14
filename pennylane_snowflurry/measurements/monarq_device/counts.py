from pennylane_snowflurry.API.api_job import Job
from pennylane.tape import QuantumTape

class Counts:
    def measure(self, tape : QuantumTape) -> dict:
        return Job(tape).run()
        