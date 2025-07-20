
import pytest
from pennylane_calculquebec.calcul_quebec_error.pennylane_cq_error import PennylaneCQError
from pennylane_calculquebec.monarq_sim import MonarqSim
from pennylane.tape import QuantumTape
import pennylane as qml

def test_monarq_sim_pennylane_cq_error(monkeypatch):
    # Patch PostProcessor.get_processor to raise PennylaneCQError with a custom message
    import pennylane_calculquebec.processing.monarq_postproc as postproc
    monkeypatch.setattr(
        postproc.PostProcessor,
        "get_processor",
        lambda *a, **kw: lambda *a2, **kw2: (_ for _ in ()).throw(PennylaneCQError("calculquebec/monarq_sim/"))
    )

    sim = MonarqSim(wires=2, shots=10)
    tape = QuantumTape()
    tape._ops = []
    tape._measurements = [qml.expval(qml.PauliZ(0))]

    with pytest.raises(PennylaneCQError) as exc_info:
        sim._measure(tape)
    # Check the error message
    assert "Error coming from Pennylane Calcul Quebec/calculquebec/monarq_sim/" in str(exc_info.value)

def test_monarq_sim_error_handling(monkeypatch):
    # Patch PostProcessor.get_processor to raise a ValueError (not ProcessingError)
    import pennylane_calculquebec.processing.monarq_postproc as postproc
    monkeypatch.setattr(postproc.PostProcessor, "get_processor", lambda *a, **kw: lambda *a2, **kw2: (_ for _ in ()).throw(ValueError("unexpected error")))

    sim = MonarqSim(wires=2, shots=10)
    tape = QuantumTape()
    tape._ops = []
    tape._measurements = [qml.expval(qml.PauliZ(0))]

    # Should propagate ValueError, not catch it
    with pytest.raises(ValueError, match="unexpected error"):
        sim._measure(tape)
