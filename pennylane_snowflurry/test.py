import numpy as np
import pennylane as qml
from pennylane_snowflurry.monarq_device import MonarqDevice
from pennylane_snowflurry.utility.test_device import TestDevice
from pennylane_snowflurry.API.client import MonarqClient
from pennylane_snowflurry.processing.config import NoPlaceNoRouteConfig
from dotenv import dotenv_values
from pennylane.devices import DefaultQubit
conf = dotenv_values(".env")

client = MonarqClient(conf["HOST"], conf["USER"], conf["ACCESS_TOKEN"], conf["PROJECT_NAME"])

transpiler_config = NoPlaceNoRouteConfig()
# transpiler_config.steps.append(ReadoutErrorMitigation())
dev_transpiler = MonarqDevice(shots=1000, client = client)
@qml.qnode(dev_transpiler)
def circuit():
    qml.RX(np.pi/2, 0)
    return qml.counts()

results = circuit()
print(results)
exit()