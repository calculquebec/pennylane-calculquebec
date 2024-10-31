from copy import deepcopy
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.transforms import transform
import pennylane_snowflurry.transpiler.base_decomposition as step1
import pennylane_snowflurry.transpiler.placement as step2
import pennylane_snowflurry.transpiler.routing as step3
import pennylane_snowflurry.transpiler.optimization as step4
import pennylane_snowflurry.transpiler.native_decomposition as step5
import pennylane_snowflurry.transpiler.transpiler_enums as enums
from pennylane_snowflurry.device_configuration import Config

class Transpiler:
    

    def get_transpiler(config : Config):
        """
        returns a transform that goes through 5 transpilation steps
        end circuit should be executable on MonarQ
        every step is optional, leaving modularity to the end user

        Args
            baseDecomposition (int) : defines how preliminary decomposition should be applied
            place (int) : defines how placing should be applied
            route (int) : defines how routing should be applied
            optimization (int) : defines how optimization should be applied
            nativeDecomposition (int) : defines how native decomposition should be applied
            use_benchmark (int) : defines how benchmark informations should be used for placing and routing
        
        all configurations for the tranpiler can be found in the TranpilerConfig class
        """
            # TODO : boolean flags could be replaced by a data structure or kwargs. 
       
       
        def transpile(tape : QuantumTape):
            """
            goes through 5 transpilation steps
            end circuit should be executable on MonarQ
            every step is optional, leaving modularity to the end user
            """
            optimized_tape = deepcopy(tape)
            with qml.QueuingManager.stop_recording():
                optimized_tape = Transpiler.base_decomp(optimized_tape, config.baseDecomposition)
                optimized_tape = Transpiler.placement(optimized_tape, config.placement, config.useBenchmark)
                optimized_tape = Transpiler.routing(optimized_tape, config.baseDecomposition, config.routing, config.useBenchmark)
                optimized_tape = Transpiler.optimize(optimized_tape, config.optimization)
                optimized_tape = Transpiler.native_decomp(optimized_tape, config.nativeDecomposition, config.optimization)
                

            new_tape = type(tape)(optimized_tape.operations, optimized_tape.measurements, shots=optimized_tape.shots)
            return [new_tape], lambda results : results[0]

        return transform(transpile)
    
    def base_decomp(tape : QuantumTape, baseDecomposition):
        if baseDecomposition == enums.BaseDecomp.CLIFFORDT: 
            return step1.base_decomposition(tape)
        return tape
    
    def placement(tape : QuantumTape, placement, benchmark):
        if placement == enums.Place.ASTAR: 
            return step2.placement_astar(tape, benchmark)
        if placement == enums.Place.VF2:
            return step2.placement_vf2(tape, benchmark)
        if placement == enums.Place.ISMAGS:
            return step2.placement_ismags(tape, benchmark)
        return tape
    
    def routing(tape : QuantumTape, baseDecomposition, routing, benchmark):
        if routing == enums.Route.ASTARSWAP:
            tape = step3.swap_routing(tape, benchmark)
            return Transpiler.base_decomp(tape, baseDecomposition)
        return tape
    
    def optimize(tape : QuantumTape, optimization):
        if optimization == enums.Optimization.COMMUTEANDMERGE:
            return step4.optimize(tape)
        return tape
    
    def native_decomp(tape : QuantumTape, nativeDecomposition, optimization):
        if nativeDecomposition == enums.NativeDecomp.MONARQ:
            optimized_tape = step5.native_gate_decomposition(tape)
            if optimization:
                optimized_tape_before = optimized_tape
                optimized_tape = step4.optimize(optimized_tape)
                optimized_tape = step5.native_gate_decomposition(optimized_tape)
                while len(optimized_tape_before.operations) > len(optimized_tape.operations):
                    optimized_tape_before = optimized_tape
                    optimized_tape = step4.optimize(optimized_tape)
                    optimized_tape = step5.native_gate_decomposition(optimized_tape)
                optimized_tape = optimized_tape_before
            return optimized_tape
        return tape
