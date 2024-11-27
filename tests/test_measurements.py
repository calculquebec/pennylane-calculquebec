import unittest.mock
from pennylane_snowflurry.transpiler.steps.placement import ISMAGS, ASTAR
from pennylane_snowflurry.transpiler.steps.routing import Swaps
import unittest
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_snowflurry.monarq_data import connectivity
import networkx as nx
from dotenv import dotenv_values
from pennylane_snowflurry.monarq_device import MonarqDevice
from pennylane_snowflurry.API.client import MonarqClient
# class test_measurements(unittest.TestCase):
    
#     def __init__(self):
#         self.conf = dotenv_values(".env")
#         self.dev = MonarqDevice(MonarqClient(self.conf["HOST"], self.conf["USER"], self.conf["ACCESS_TOKEN"]))
    
#     @unittest.mock.patch("pennylane_snowflurry.utility.graph_utility.machine_graph")
#     def test_place_no_4(self, machine_graph):
#         pass
    