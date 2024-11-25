from copy import deepcopy
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.transforms import transform
from pennylane_snowflurry.transpiler.transpiler_config import TranspilerConfig
from pennylane_snowflurry.transpiler.steps.interfaces.post_processing import PostProcStep

class PostProcessor:
    

    def get_processor(behaviour_config : TranspilerConfig):
        def process(tape : QuantumTape, results : dict[str, int]):
            """
            Args:
                tape (QuantumTape) : the tape for which the results were calculated
                results (dict[str, int]) : the results you want to process
            
            Returns : 
                The processed results
            """
            expanded_tape = PostProcessor.expand_full_measurements(tape)
            
            postproc_steps = [step for step in behaviour_config.steps if isinstance(step, PostProcStep)]
            processed_results = deepcopy(results)
            for step in postproc_steps:
                processed_results = step.execute(expanded_tape, processed_results)
            return processed_results

        return process

    def expand_full_measurements(tape):
        mps = []
        for mp in tape.measurements:
            if mp.wires == None or len(mp.wires) < 1:
                mps.append(type(mp)(wires=tape.wires))
            else:
                mps.append(mp)
        
        return type(tape)(tape.operations, mps, shots=tape.shots)
            