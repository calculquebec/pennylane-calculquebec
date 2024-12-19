# This file is used to run the tests developed in the Pennylane library.
# The tests are run using the pytest framework and can be run using the following command:
# python-jl -m tests.test_pennylaneTests

# TOFIX : This test doesn't work.
from pennylane.devices.tests import test_device
import pytest

@pytest.mark.xfail
def test_pennylane_devices():
    test_device("snowflurry.qubit", skip_ops=True)
