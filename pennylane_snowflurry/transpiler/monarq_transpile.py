from copy import deepcopy
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.transforms import transform
from pennylane_snowflurry.transpiler.config import TranspilerConfig
from pennylane_snowflurry.transpiler.interfaces import PreProcStep
class PreProcessor:
    

    def get_processor(behaviour_config : TranspilerConfig, circuit_wires):
        """
        returns a transform that goes through given transpilation steps\n
        every step is optional and new steps can be added, leaving modularity to the end user\n

        Args\n
            config (Config) : defines which transpilation steps you want to run on your code\n
                Default value applies those steps : \n
                    1. decomposition to clifford + t set\n
                    2. placement using a pathfinding heuristic\n
                    3. routing using swaps\n
                    4. optimization using commutations, merges and cancellations of inverses and trivial gates\n
                    5. decomposition to MonarQ's native gate set\n
        """
        def transpile(tape : QuantumTape):
            """
            Args:
                tape (QuantumTape) : the tape you want to transpile
            
            Returns : 
                A transform dispatcher object that can be used in the preprocess method of pennylane Devices
            """
            wires = tape.wires if circuit_wires is None or len(tape.wires) > len(circuit_wires) else circuit_wires
            optimized_tape = PreProcessor.expand_full_measurements(tape, wires)
            
            with qml.QueuingManager.stop_recording():
                prerpoc_steps = [step for step in behaviour_config.steps if isinstance(step, PreProcStep)]
                for step in prerpoc_steps:
                    optimized_tape = step.execute(optimized_tape)
            new_tape = type(tape)(optimized_tape.operations, optimized_tape.measurements, shots=optimized_tape.shots)
            return [new_tape], lambda results : results[0]

        return transform(transpile)

    def expand_full_measurements(tape, wires):
        mps = []
        for mp in tape.measurements:
            if mp.wires == None or len(mp.wires) < 1:
                mps.append(type(mp)(wires=wires))
            else:
                mps.append(mp)
        
        return type(tape)(tape.operations, mps, shots=tape.shots)
            