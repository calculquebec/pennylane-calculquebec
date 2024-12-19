from pennylane_calculquebec.processing.config import ProcessingConfig, MonarqDefaultConfig, MonarqDefaultConfigNoBenchmark, NoPlaceNoRouteConfig, EmptyConfig, FakeMonarqConfig
from pennylane_calculquebec.processing.steps import DecomposeReadout, CliffordTDecomposition, ASTAR, ISMAGS, Swaps, IterativeCommuteAndMerge, MonarqDecomposition, GateNoiseSimulation, ReadoutNoiseSimulation

def test_processing_config():
    config = ProcessingConfig(1, 2, 3)
    assert all(a in config.steps for a in [1,2,3])
    assert len(config.steps) == 3

def test_presets():
    # default config should contain only default steps
    config = MonarqDefaultConfig()
    test_arr = [DecomposeReadout, CliffordTDecomposition, ISMAGS, Swaps, IterativeCommuteAndMerge, MonarqDecomposition]
    for step in config.steps:
        assert any(type(step) == test for test in test_arr)
    
    # benchmarking steps should not use benchmarks
    config = MonarqDefaultConfigNoBenchmark()
    benchmark_steps = filter(lambda step : hasattr(step, "use_benchmark"), config.steps)
    
    assert all(not step.use_benchmark for step in benchmark_steps)
    
    # no place no route config should not contain placement or routing
    config = NoPlaceNoRouteConfig()
    
    place_route = list(filter(lambda step : isinstance(step, ASTAR) or isinstance(step, Swaps), config.steps))
    assert len(place_route) == 0
    
    # empty config should be empty
    config = EmptyConfig()
    assert len(config.steps) == 0
    
    # all default steps should also be in fake config
    config = FakeMonarqConfig()
    default = MonarqDefaultConfig()
    assert len(config.steps) == len(default.steps) + 2
    
    for step in default.steps:
        assert any(type(def_step) == type(step) for def_step in config.steps)