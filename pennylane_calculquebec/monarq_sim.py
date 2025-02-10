"""
Contains a wrapper around default.mixed which uses MonarQ pre/post processing\n
"""

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_calculquebec.processing.monarq_postproc import PostProcessor
from pennylane_calculquebec.processing.config import FakeMonarqConfig
from pennylane.measurements import CountsMP
from pennylane_calculquebec.device_exception import DeviceException
from pennylane_calculquebec.base_device import BaseDevice


class MonarqSim(BaseDevice):
    """
    a device that uses the monarq transpiler but simulates results using default.mixed
    """
    name = "MonarqSim"
    short_name = "monarq.sim"
    pennylane_requires = ">=0.30.0"
    author = "CalculQuebec"
    
    realm = "calculqc"
    circuit_name = "test circuit"
    project_id = ""
    machine_name = "yamaska"

    observables = {
        "PauliZ"
    }

    @property
    def name(self):
        return MonarqSim.short_name
    
    def _measure(self, tape : QuantumTape):
        """
        simulates job to Monarq and returns value, converted to required measurement type

        Args : 
            tape (QuantumTape) : the tape from which to get results
        
        Returns :
            a result, which format can change according to the measurement process
        """
        if len(tape.measurements) != 1:
            raise DeviceException("Multiple measurements not supported")
        meas = type(tape.measurements[0]).__name__

        if not any(meas == measurement for measurement in MonarqSim.measurement_methods.keys()):
            raise DeviceException("Measurement not supported")

        # simulate counts from given circuit on default mixed
        counts_tape = type(tape)(ops=tape.operations, 
                                measurements=[CountsMP(wires=mp.wires) for mp in tape.measurements],
                                shots=1000)
        results = qml.execute([counts_tape], qml.device("default.mixed", wires = tape.wires))[0]

        # apply post processing
        results = PostProcessor.get_processor(self._processing_config, self.wires)(counts_tape, results)

        # return desired measurement method
        measurement_method = MonarqSim.measurement_methods[meas]
        return measurement_method(results)

    @property
    def default_processing_config(self):
        return FakeMonarqConfig()