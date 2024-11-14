from copy import deepcopy
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.transforms import transform
import pennylane_snowflurry.transpiler.steps.base_decomposition as step1
import pennylane_snowflurry.transpiler.steps.placement as step2
import pennylane_snowflurry.transpiler.steps.routing as step3
import pennylane_snowflurry.transpiler.steps.optimization as step4
import pennylane_snowflurry.transpiler.steps.native_decomposition as step5
from pennylane_snowflurry.transpiler.transpiler_config import TranspilerConfig

class Transpiler:
    

    def get_transpiler(behaviour_config : TranspilerConfig):
        """
        returns a transform that goes through given transpilation steps\n
        every step is optional and new steps can be added, leaving modularity to the end user\n

        Args\n
            config (Config) : contains which transpilation steps you want to run on your code\n
                Default value contains those steps : \n
                    1. decomposition to clifford + t set\n
                    2. placement using a pathfinding heuristic\n
                    3. routing using swaps\n
                    4. optimization using commutations, merges and cancellations of inverses and trivial gates\n
                    5. decomposition to MonarQ's native gate set\n
        """
        def transpile(tape : QuantumTape):
            """
            goes through 5 transpilation steps
            end circuit should be executable on MonarQ
            every step is optional, leaving modularity to the end user
            """
            optimized_tape = deepcopy(tape)
            with qml.QueuingManager.stop_recording():
                for step in behaviour_config.steps:
                    optimized_tape = step.execute(optimized_tape)
            new_tape = type(tape)(optimized_tape.operations, optimized_tape.measurements, shots=optimized_tape.shots)
            return [new_tape], lambda results : results[0]

        return transform(transpile)
    