from pennylane_calculquebec.processing.interfaces import PreProcStep, PostProcStep

def test_pre_proc_step():
    step = PreProcStep()
    result = step.execute("this is a test")
    assert result == "this is a test"

def test_post_proc_step():
    step = PostProcStep()
    result = step.execute("this should not return", "this should return")
    assert result == "this should return"