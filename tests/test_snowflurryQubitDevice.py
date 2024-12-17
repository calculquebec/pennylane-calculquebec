import pytest
from pennylane_calculquebec.snowflurry_device import SnowflurryQubitDevice
from juliacall import newmodule


class TestSnowflurryQubitDevice:

    @pytest.fixture(autouse=True)
    def sf_namespace(self):
        self.snowflurry = newmodule("Snowflurry")
        self.snowflurry.seval("using Snowflurry")

    def test_initialization(self):
        # Example parameters
        num_wires = 4
        num_shots = 1000
        seed = 42

        # Create an instance of the SnowflurryQubitDevice
        device = SnowflurryQubitDevice(wires=num_wires, shots=num_shots)

        assert device.num_wires == num_wires

        assert device.shots.total_shots == num_shots

    @pytest.mark.xfail
    def test_hadamard_juliacall(self):
        self.snowflurry.c = self.snowflurry.QuantumCircuit(qubit_count=3)
        self.snowflurry.seval("push!(c,hadamard(1))")

