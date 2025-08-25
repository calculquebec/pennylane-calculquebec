import pytest
import pennylane_calculquebec.utility.debug as debug


def test_probs_to_counts_basic_two_outcomes():
    # For a length-2 probability vector, we expect padding to 2 bits ("00", "01")
    probs = [0.5, 0.5]
    shots = 1000
    result = debug.probs_to_counts(probs, shots)
    assert result == {"00": 500, "01": 500}
    assert sum(result.values()) == shots


def test_probs_to_counts_basic_four_outcomes():
    probs = [0.25, 0.25, 0.25, 0.25]
    shots = 400
    result = debug.probs_to_counts(probs, shots)
    # bit length -> log2(4)=2 +1 => 3 bits => labels 000..011
    assert result == {"000": 100, "001": 100, "010": 100, "011": 100}
    assert sum(result.values()) == shots


def test_probs_to_counts_invalid_length():
    # length not power of two
    with pytest.raises(ValueError):
        debug.probs_to_counts([0.2, 0.3, 0.5], 10)


def test_probs_to_counts_empty():
    with pytest.raises(ValueError):
        debug.probs_to_counts([], 10)
