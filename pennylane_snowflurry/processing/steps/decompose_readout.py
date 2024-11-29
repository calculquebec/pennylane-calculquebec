from pennylane_snowflurry.processing.interfaces import PreProcStep
from pennylane.tape import QuantumTape
import pennylane as qml
import numpy as np
class DecomposeReadout(PreProcStep):
    
    def get_ops_for_product(self, obs):
        ops = []
        for op in obs.operands:
            if op.name in self.obs_dict:
                for w in op.wires:
                    ops.append(self.obs_dict[op.name](w))
                continue
            
            if op.name == "Prod":
                ops.extend(self.get_ops_for_product(op))
                continue
            
            raise ValueError("this readout observable is not supported")
        return ops
        
    @property
    def obs_dict(self):
        return {
            "PauliZ" : lambda w : qml.Identity(w),
            "PauliX" : lambda w : qml.RY(np.pi / 2, w),
            "PauliY" : lambda w : qml.RX(-np.pi / 2, w),
            "Hadamard" : lambda w : qml.RY(np.pi / 4, w)
        }
        
    def execute(self, tape : QuantumTape):
        ops = tape.operations.copy()
        mps = []
        for mp in tape.measurements:
            # if there is no obs, skip
            if mp.obs is None:
                mps.append(mp)
                continue
            
            # if op is supported, apply rotation and change mp's observable to Z
            if mp.obs.name in self.obs_dict:
                wires = [w for w in mp.obs.wires]
                for w in wires:
                    ops.append(self.obs_dict[mp.obs.name](w))
                mps.append(type(mp)(wires=wires))
                continue
            
            # if op is a product, get the list of rotations that represent this product, and change mp's observable to Z
            if mp.obs.name == "Prod":
                wires = [w for w in mp.obs.wires]
                for op in self.get_ops_for_product(mp.obs):
                    ops.append(op)
                mps.append(type(mp)(wires=wires))
                continue
                
                
            # if we reach this point, it means that we can't readout on this observable
            raise ValueError("this readout observable is not supported")
        
        return type(tape)(ops, mps, shots=tape.shots)
            