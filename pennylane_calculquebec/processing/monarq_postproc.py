"""
Contains a processor class for post-processing steps
"""

from copy import deepcopy
from pennylane.tape import QuantumTape
from pennylane_calculquebec.processing.config import ProcessingConfig
from pennylane_calculquebec.processing.interfaces import PostProcStep

class PostProcessor:
    """
    a container for post-processing functionalities that should be applied to the results of a circuit
    """
    @staticmethod
    def get_processor(behaviour_config : ProcessingConfig, circuit_wires):
        """
        returns a function that applies the steps contained in the supplied ProcessingConfig

        Args:
            behaviour_config (ProcessingConfig): a processing config to apply
            circuit_wires (list[int]): the wires in the circuit
        """
        def process(tape : QuantumTape, results : dict[str, int]):
            """
            applies a list of post-processing steps
            
            Args:
                tape (QuantumTape) : the tape for which the results were calculated
                results (dict[str, int]) : the results you want to process
            
            Returns : 
                The processed results
            """
            wires = tape.wires if circuit_wires is None or len(tape.wires) > len(circuit_wires) else circuit_wires
            expanded_tape = PostProcessor.expand_full_measurements(tape, wires)
            
            postproc_steps = [step for step in behaviour_config.steps if isinstance(step, PostProcStep)]
            processed_results = deepcopy(results)
            for step in postproc_steps:
                processed_results = step.execute(expanded_tape, processed_results)
            return processed_results

        return process

    @staticmethod
    def expand_full_measurements(tape, wires):
        """turns empty measurements to all-wire measurements

        Args:
            tape (QuantumTape): the quantum tape from which to expand the measurements
            wires (list[int]): wires from the circuit

        Returns:
            QuantumTape: transformed tape 
        """
        mps = []
        for mp in tape.measurements:
            if mp.wires == None or len(mp.wires) < 1:
                mps.append(type(mp)(wires=wires))
            else:
                mps.append(mp)
        
        return type(tape)(tape.operations, mps, shots=tape.shots)
            