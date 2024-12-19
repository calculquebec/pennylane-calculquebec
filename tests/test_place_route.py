from pennylane_calculquebec.processing.steps.placement import ISMAGS, ASTAR
from pennylane_calculquebec.processing.steps.routing import Swaps
from unittest.mock import patch
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_calculquebec.monarq_data import connectivity
import networkx as nx
import pytest
from pennylane.wires import Wires
