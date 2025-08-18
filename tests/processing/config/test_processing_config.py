from pennylane_calculquebec.processing.config import *
from pennylane_calculquebec.processing.steps import *


class Step:
    def __init__(self, arg):
        self.arg = arg


def test_processing_config():
    steps = [Step(1), Step(2), Step(3)]
    config = ProcessingConfig(*steps)

    assert all(a in config.steps for a in steps)
    assert len(config.steps) == 3

    expected_config = ProcessingConfig(Step(1), Step(2))
    assert config != expected_config

    expected_config = ProcessingConfig(Step(1), Step(2), Step(4))
    assert config != expected_config

    expected_config = ProcessingConfig(Step(1), Step(2), Step([3]))
    assert config != expected_config

    expected_config = ProcessingConfig(Step(1), Step(2), Step(3))
    assert config == expected_config

    assert config[0] == steps[0]

    step = Step(4)
    config[0] = step
    assert config.steps[0] == step


def test_presets():
    # default config should contain only default steps
    config = MonarqDefaultConfig("yamaska")
    test_arr = [
        DecomposeReadout,
        CliffordTDecomposition,
        VF2,
        Swaps,
        IterativeCommuteAndMerge,
        MonarqDecomposition,
    ]
    for step in config.steps:
        assert any(type(step) == test for test in test_arr)

    # benchmarking steps should not use benchmarks
    config = MonarqDefaultConfigNoBenchmark("yamaska")
    benchmark_steps = filter(lambda step: hasattr(step, "use_benchmark"), config.steps)

    assert all(not step.use_benchmark for step in benchmark_steps)

    # no place no route config should not contain placement or routing
    config = NoPlaceNoRouteConfig()

    place_route = list(
        filter(
            lambda step: isinstance(step, VF2) or isinstance(step, Swaps), config.steps
        )
    )
    assert len(place_route) == 0

    # empty config should be empty
    config = EmptyConfig()
    assert len(config.steps) == 0

    # all default steps should also be in fake config
    config = FakeMonarqConfig("yamaska")
    default = MonarqDefaultConfig("yamaska")
    assert len(config.steps) == len(default.steps) + 2

    for step in default.steps:
        assert any(type(def_step) == type(step) for def_step in config.steps)
    assert any(
        type(def_step) == type(GateNoiseSimulation(False)) for def_step in config.steps
    )
    assert any(
        type(def_step) == type(ReadoutNoiseSimulation(False))
        for def_step in config.steps
    )

    # print default config is the same as default + prints at start and end
    config = PrintDefaultConfig("yamaska")
    for step in default.steps:
        assert any(type(def_step) == type(step) for def_step in config.steps)
    assert any(type(def_step) == type(PrintWires()) for def_step in config.steps)

    config = PrintDefaultConfig("yamaska", False)
    assert any(type(def_step) == type(PrintTape()) for def_step in config.steps)

    # print no place no route config is the same as default + prints at start and end
    config = PrintNoPlaceNoRouteConfig()
    no_place_no_route = NoPlaceNoRouteConfig()
    for step in no_place_no_route.steps:
        assert any(type(def_step) == type(step) for def_step in config.steps)
    assert any(type(def_step) == type(PrintWires()) for def_step in config.steps)

    config = PrintNoPlaceNoRouteConfig(False)
    assert any(type(def_step) == type(PrintTape()) for def_step in config.steps)
