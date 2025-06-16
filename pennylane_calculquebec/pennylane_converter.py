import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result
from pennylane.measurements import (
    StateMeasurement,
    MeasurementProcess,
    MeasurementValue,
    ProbabilityMP,
    SampleMP,
    ExpectationMP,
    CountsMP,
    StateMP,
)
import time
import re
from pennylane.typing import TensorLike
from typing import Callable, Type
import importlib.util
from pennylane_calculquebec.measurements import (
    MeasurementStrategy,
    Sample,
    Counts,
    Probabilities,
    ExpectationValue,
    State
)
from pennylane_calculquebec.logger import logger


if importlib.util.find_spec("juliacall") is None:
    raise Exception("""
    juliacall is required.
    Please install the extra julia in order to import snowflurry_device : pip install --no-index pennylane-calculquebec[julia]
    """)


# Dictionary mapping PennyLane operations to Snowflurry operations
# The available Snowflurry operations are listed here:
# https://snowflurrysdk.github.io/Snowflurry.jl/dev/library/quantum_toolkit.html
# https://snowflurrysdk.github.io/Snowflurry.jl/dev/library/quantum_gates.html
# https://snowflurrysdk.github.io/Snowflurry.jl/dev/library/quantum_circuit.html
SNOWFLURRY_OPERATION_MAP = {
    "PauliX": "sigma_x({0})",
    "PauliY": "sigma_y({0})",
    "PauliZ": "sigma_z({0})",
    "Hadamard": "hadamard({0})",
    "CNOT": "control_x({0},{1})",
    "CY": "controlled(sigma_y({1}),[{0}])",  # 0 is the control qubit, 1 is the target qubit
    "CZ": "control_z({0},{1})",
    "SWAP": "swap({0},{1})",
    "ISWAP": "iswap({0},{1})",
    "RX": "rotation_x({1},{0})",
    "RY": "rotation_y({1},{0})",
    "RZ": "rotation_z({1},{0})",
    "Identity": "identity_gate({0})",
    "CSWAP": "controlled(swap({1},{2}),[{0}])",  # 0 is the control qubit, 1 and 2 are the target qubits
    "CRX": "controlled(rotation_x({2},{0}),[{1}])",  # 0 is the angle, 1 is the control qubit, 2 is the target qubit
    "CRY": "controlled(rotation_y({2},{0}),[{1}])",
    "CRZ": "controlled(rotation_z({2},{0}),[{1}])",
    "PhaseShift": "phase_shift({1},{0})",  # 0 is the angle, 1 is the wire
    "ControlledPhaseShift": "controlled(phase_shift({2},{0}),[{1}])",
    # 0 is the angle, 1 is the control qubit, 2 is the target qubit
    "Toffoli": "toffoli({0},{1},{2})",
    "U3": "universal({3},{0},{1},{2})",  # 3 is the wire, 0,1,2 are theta, phi, delta respectively
    "T": "pi_8({0})",
    "Rot": "rotation({3},{0},{1})",  # theta, phi but no omega so we skip {2}, {3} is the wire
}


"""
if host, user, access_token are left blank, the code will be ran on the simulator
if host, user, access_token are filled, the code will be sent to Anyon's API
"""

def define_snowflurry_namespace():
    """
    Define the Snowflurry namespace in Julia.
    """
    from juliacall import newmodule
    sf = newmodule("Snowflurry")
    sf.seval("using Snowflurry")
    return sf


class PennylaneConverter:
    """
    A PennyLane converter for the Snowflurry device.

    Is in charge of interfacing with the Snowflurry.jl package and converting PennyLane circuits to
    Snowflurry circuits.
    """

    ###################################
    # Class attributes used for logic #
    ###################################
    snowflurry_readout_name = "Readout"
    snowflurry_gate_object_name = "Gate Object"

    # Pattern is found in PyCall.jlwrap object of Snowflurry.QuantumCircuit.instructions
    snowflurry_str_search_pattern = r"Gate Object: (.*)\n"

    #################
    # Class methods #
    #################

    def __init__(
        self,
        pennylane_circuit: QuantumTape,
        debugger=None,
        interface=None,
        host="",
        user="",
        access_token="",
        project_id="",
        realm="",
        wires=None
    ):
        try:
            # Instance attributes related to PennyLane
            self.pennylane_circuit = pennylane_circuit
            self.debugger = debugger
            self.interface = interface
            self.wires = wires

            # Instance attributes related to Snowflurry
            self.snowflurry_namespace = define_snowflurry_namespace()
            self.snowflurry_py_circuit = None
            if (
                len(host) != 0
                and len(user) != 0
                and len(access_token) != 0
                and len(realm) != 0
            ):
                self.snowflurry_namespace.currentClient = self.snowflurry_namespace.Client(
                    host=host, user=user, access_token=access_token, realm=realm
                )
                self.snowflurry_namespace.seval('project_id="' + project_id + '"')
            else:
                self.snowflurry_namespace.currentClient = None

            self.measurementStrategy = None
        except Exception as e:
            logger.error("Error %s in __init__ located in PennylaneConverter: %s", type(e).__name__, e)

    def simulate(self):
        try:
            self.snowflurry_py_circuit = self.convert_circuit(
                self.pennylane_circuit
            )
            return self.measure_final_state()
        except Exception as e:
            logger.error("Error %s in simulate located in PennylaneConverter: %s", type(e).__name__, e)
            return None

    def convert_circuit(
        self, pennylane_circuit: QuantumTape,
    ):
        try:
            wires_nb = self.wires  # default number of wires in the circuit
            self.snowflurry_namespace.sf_circuit = self.snowflurry_namespace.QuantumCircuit(qubit_count=wires_nb)

            prep = None
            if len(pennylane_circuit) > 0 and isinstance(
                pennylane_circuit[0], qml.operation.StatePrepBase
            ):
                prep = pennylane_circuit[0]

            # Add gates to Snowflurry circuit
            for op in pennylane_circuit.operations[bool(prep):]:
                if op.name in SNOWFLURRY_OPERATION_MAP:
                    if SNOWFLURRY_OPERATION_MAP[op.name] == NotImplementedError:
                        logger.warning("%s is not implemented yet, skipping...", op.name)
                        continue
                    parameters = op.parameters + [i + 1 for i in op.wires.tolist()]
                    gate = SNOWFLURRY_OPERATION_MAP[op.name].format(*parameters)
                    self.snowflurry_namespace.seval(f"push!(sf_circuit,{gate})")
                else:
                    logger.warning("%s is not supported by this device. skipping...", op.name)

            return self.snowflurry_namespace.sf_circuit
        except Exception as e:
            logger.error("Error %s in convert_circuit located in PennylaneConverter: %s", type(e).__name__, e)
            return None

    def apply_readouts(self, obs):
        try:
            if obs is None:  # if no observable is given, we apply readouts to all wires
                for wire in range(self.wires):
                    self.snowflurry_namespace.seval(f"push!(sf_circuit, readout({wire + 1}, {wire + 1}))")
            else:
                # if an observable is given, we apply readouts to the wires mentioned in the observable,
                # TODO : could add Pauli rotations to get the correct observable
                self.apply_single_readout(obs.wires[0])
        except Exception as e:
            logger.error("Error %s in apply_readouts located in PennylaneConverter: %s", type(e).__name__, e)

    def get_circuit_as_dictionary(self):
        try:
            ops = []
            instructions = (
                self.snowflurry_namespace.namespace.sf_circuit.instructions
            )  # instructions is a jlwrap object
            gate_str = ""
            gate_name = ""

            for inst in instructions:

                gate_str = str(inst)  # convert the jlwrap object to a string

                try:
                    if self.snowflurry_gate_object_name in gate_str:
                        # if the gate is a Gate object, we extract the name and the connected qubits
                        # from the string with a regex
                        gate_name = re.search(
                            self.snowflurry_str_search_pattern, gate_str
                        ).group(1)
                        op_data = {
                            "gate": gate_name,
                            "connected_qubits": list(inst.connected_qubits),
                        }
                    if self.snowflurry_readout_name in gate_str:
                        # if the gate is a Readout object, we extract the connected qubit from the string
                        gate_name = self.snowflurry_readout_name
                        op_data = {
                            "gate": gate_name,
                            "connected_qubits": [inst.connected_qubit],
                        }
                except Exception as e_inner:
                    logger.error("Error %s in get_circuit_as_dictionary (parsing %s) located in PennylaneConverter: %s", type(e_inner).__name__, gate_str, e_inner)
                    raise ValueError(f"Error while parsing {gate_str}: {e_inner}")

                ops.append(op_data)

            return ops
        except Exception as e:
            logger.error("Error %s in get_circuit_as_dictionary located in PennylaneConverter: %s", type(e).__name__, e)
            return []

    def has_readout(self) -> bool:
        try:
            ops = self.get_circuit_as_dictionary()
            for op in ops:
                if op["gate"] == self.snowflurry_readout_name:
                    return True
            return False
        except Exception as e:
            logger.error("Error %s in has_readout located in PennylaneConverter: %s", type(e).__name__, e)
            return False

    def remove_readouts(self):
        try:
            while self.has_readout():
                self.snowflurry_namespace.namespace.seval("pop!(sf_circuit)")
        except Exception as e:
            logger.error("Error %s in remove_readouts located in PennylaneConverter: %s", type(e).__name__, e)

    def apply_single_readout(self, wire):
        try:
            ops = self.get_circuit_as_dictionary()

            for op in ops:
                if op["gate"] == self.snowflurry_readout_name:
                    if op["connected_qubits"] == wire - 1:  # wire is 1-indexed in Julia
                        return

            self.snowflurry_namespace.namespace.seval(f"push!(sf_circuit, readout({wire+1}, {wire+1}))")
        except Exception as e:
            logger.error("Error %s in apply_single_readout located in PennylaneConverter: %s", type(e).__name__, e)

    def measure_final_state(self):
        try:
            circuit = self.pennylane_circuit.map_to_standard_wires()
            shots = circuit.shots.total_shots
            if shots is None:
                shots = 1

            if len(circuit.measurements) == 1:
                results = self.measure(
                    circuit.measurements[0], shots
                )
            else:
                results = tuple(
                    self.measure(mp, shots)
                    for mp in circuit.measurements
                )

            return results
        except Exception as e:
            logger.error("Error %s in measure_final_state located in PennylaneConverter: %s", type(e).__name__, e)
            return None

    def measure(self, mp: MeasurementProcess, shots):
        try:
            self.measurementStrategy = self.get_strategy(mp)
            result = self.measurementStrategy.measure(self, mp, shots)
            return result
        except Exception as e:
            logger.error("Error %s in measure located in PennylaneConverter: %s", type(e).__name__, e)
            return None

    def get_strategy(self, mp: MeasurementProcess):
        try:
            if isinstance(mp, CountsMP):
                return Counts()
            elif isinstance(mp, SampleMP):
                return Sample()
            elif isinstance(mp, ProbabilityMP):
                return Probabilities()
            elif isinstance(mp, ExpectationMP):
                return ExpectationValue()
            elif isinstance(mp, StateMP):
                return State()
            else:
                raise ValueError(f"Measurement process {mp} is not supported by this device.")
        except Exception as e:
            logger.error("Error %s in get_strategy located in PennylaneConverter: %s", type(e).__name__, e)
            raise
