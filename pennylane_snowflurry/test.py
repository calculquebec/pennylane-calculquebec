import numpy as np
import pennylane as qml
import pennylane_snowflurry.utility.test_circuits as test_circuits
from pennylane_snowflurry.monarq_device import MonarqDevice
from pennylane_snowflurry.utility.test_device import TestDevice
from pennylane_snowflurry.API.client import MonarqClient
from pennylane_snowflurry.transpiler.config import MonarqDefaultConfig
from pennylane_snowflurry.transpiler.steps import ReadoutErrorMitigation
from dotenv import dotenv_values
from pennylane_snowflurry.utility.debug import arbitrary_circuit

conf = dotenv_values(".env")

client = MonarqClient(conf["HOST"], conf["USER"], conf["ACCESS_TOKEN"], conf["PROJECT_NAME"])

transpiler_config = MonarqDefaultConfig()
transpiler_config.steps.append(ReadoutErrorMitigation())
dev_transpiler = MonarqDevice(wires=[0, 1, 2], shots=1000, client=client)

@qml.qnode(dev_transpiler)
def circuit():
    qml.X(0)
    return qml.counts()
results = circuit()
print(results)
exit()
for i in range(8):
    node = qml.QNode(lambda : test_circuits.bernstein_vazirani(i, 4), dev_transpiler)
    print(node())
    node = qml.QNode(lambda : arbitrary_circuit(node.tape), dev_transpiler)
