import pytest
from unittest.mock import patch
from pennylane_calculquebec.processing import PreProcessor
from pennylane_calculquebec.processing.interfaces import PreProcStep, PostProcStep

@pytest.fixture
def mock_expand_full_measurements():
    with patch("pennylane_calculquebec.processing.PreProcessor.expand_full_measurements") as mock:
        yield mock

class step_call_counter:
    def __init__(self):
        self.i = 0
        
class op:
    def __init__(self, wires):
        self.wires = wires
        
class Tape:
    def __init__(self, ops = [], mps = [], wires = [], shots = None):
        self.operations = ops
        self.measurements = mps
        self.wires = wires
        self.shots = shots
        self.results = []

class config:
    def __init__(self, *params):
        self.steps = [p for p in params]
        
class step(PreProcStep):
    def __init__(self, test, call_counter):
        self.test = test
        self.call_counter = call_counter
        
    def execute(self, tape):
        self.call_counter.i += 1
        tape.results += [self.test]
        return tape
        

def test_get_processor(mock_expand_full_measurements):
    call_counter = step_call_counter()
    conf = config(step("a", call_counter), 
                  step("b", call_counter), 
                  PostProcStep(),
                  step("c", call_counter), 
                  "not_step")
    tape = Tape()
    process = PreProcessor.get_processor(conf, [0, 1, 2])
    tape2 = process(tape)[0][0]
    
    solution = ["a", "b", "c"]
    
    mock_expand_full_measurements.assert_called_once_with(tape, [0, 1, 2])
    assert call_counter.i == 3
    for i, r in enumerate(tape2.results):
        assert solution[i] == r

def test_expand_full_measurements():
    tape = Tape(mps = [op([])])
    result : Tape = PreProcessor.expand_full_measurements(tape, [4,1,2])
    solution = [4,1,2]
    for i, w in enumerate(solution):
        assert w == result.measurements[0].wires[i]
    
    tape = Tape(mps = [op([1,2])])
    result : Tape = PreProcessor.expand_full_measurements(tape, [4,1,2])
    solution = [1,2]
    for i, w in enumerate(solution):
        assert w == result.measurements[0].wires[i]
        
    tape = Tape(mps = [op([1]), op([2])])
    result : Tape = PreProcessor.expand_full_measurements(tape, [4,1,2])
    solution = [1, 2]
    for i, w in enumerate(solution):
        assert w == result.measurements[i].wires[0]
        