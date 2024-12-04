import numpy as np
import pennylane as qml
from pennylane_snowflurry.monarq_device import MonarqDevice
from pennylane_snowflurry.utility.test_device import TestDevice
from pennylane_snowflurry.API.client import MonarqClient
from pennylane_snowflurry.processing.config import NoPlaceNoRouteConfig, MonarqDefaultConfig
from pennylane_snowflurry.processing.steps import ReadoutErrorMitigation, ISMAGS

from pennylane.devices import DefaultQubit
from pennylane_snowflurry.API.adapter import ApiAdapter
from pennylane_snowflurry.utility.graph import calculate_cost, machine_graph, keys
from pennylane_snowflurry.utility.plotting import graph
import matplotlib.pyplot as plt
from dotenv import dotenv_values
conf = dotenv_values(".env")

client = MonarqClient(conf["HOST"], conf["USER"], conf["ACCESS_TOKEN"], conf["PROJECT_NAME"])

transpiler_config = MonarqDefaultConfig()
dev_transpiler = MonarqDevice(shots=1000, client = client, processing_config=transpiler_config)

for i in range(1, 8):
    
    @qml.qnode(dev_transpiler)
    def circuit():
        qml.Hadamard(0)
        for j in range(i):
            qml.CNOT([0, j+1])
        return qml.counts()

    results = circuit()
    print(results)
    plt.bar([v for v in results.keys()], [v for v in results.values()])
    plt.show()
            
# expliquer la différence entre les portes qml et custom
# exit()