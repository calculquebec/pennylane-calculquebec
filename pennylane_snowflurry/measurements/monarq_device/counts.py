from pennylane_snowflurry.API.job import Job
from pennylane.tape import QuantumTape

class Counts:
    """
    a count measurement class for MonarqDevice
    
    TODO : could inherit from MeasurementStrategy once the latter does not depend on PennylaneConverter
    """
    def measure(self, tape : QuantumTape) -> dict:
        return Job(tape).run()
        