import pytest
from unittest.mock import patch
from pennylane_calculquebec.processing import PostProcessor
from pennylane_calculquebec.processing.interfaces import PostProcStep, PreProcStep

@pytest.fixture
def mock_expand_full_measurements():
    with patch("pennylane_calculquebec.processing.PostProcessor.expand_full_measurements") as mock:
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

class config:
    def __init__(self, *params):
        self.steps = [p for p in params]
        
class step(PostProcStep):
    def __init__(self, test, call_counter):
        self.test = test
        self.call_counter = call_counter
        
    def execute(self, tape, results):
        self.call_counter.i += 1
        return results + [self.test]
        

def test_get_processor(mock_expand_full_measurements):
    call_counter = step_call_counter()
    conf = config(step("a", call_counter), 
                  step("b", call_counter), 
                  PreProcStep(),
                  step("c", call_counter), 
                  "not_step")
    tape = Tape()
    process = PostProcessor.get_processor(conf, [0, 1, 2])
    result = process(tape, [])
    
    solution = ["a", "b", "c"]
    
    mock_expand_full_measurements.assert_called_once_with(tape, [0,1,2])
    assert call_counter.i == 3
    for i, r in enumerate(result):
        assert solution[i] == r

def test_expand_full_measurements():
    tape = Tape(mps = [op([])])
    result : Tape = PostProcessor.expand_full_measurements(tape, [4,1,2])
    solution = [4,1,2]
    for i, w in enumerate(solution):
        assert w == result.measurements[0].wires[i]
    
    tape = Tape(mps = [op([1,2])])
    result : Tape = PostProcessor.expand_full_measurements(tape, [4,1,2])
    solution = [1,2]
    for i, w in enumerate(solution):
        assert w == result.measurements[0].wires[i]
        
    tape = Tape(mps = [op([1]), op([2])])
    result : Tape = PostProcessor.expand_full_measurements(tape, [4,1,2])
    solution = [1, 2]
    for i, w in enumerate(solution):
        assert w == result.measurements[i].wires[0]
        