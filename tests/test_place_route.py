from pennylane_snowflurry.processing.steps.placement import ISMAGS, ASTAR
from pennylane_snowflurry.processing.steps.routing import Swaps
from unittest.mock import patch
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_snowflurry.monarq_data import connectivity
import networkx as nx
import pytest
from pennylane.wires import Wires
