from .measurement_strategy import MeasurementStrategy
from collections import Counter
from pennylane_calculquebec.calcul_quebec_error import measurement_error
from pennylane_calculquebec.logger import logger
class Counts(MeasurementStrategy):

    def __init__(self):
        super().__init__()

    def measure(self, converter, mp, shots):
        if self.Snowflurry.currentClient is None:
            converter.remove_readouts()
            converter.apply_readouts(mp.obs)
            shots_results = self.Snowflurry.simulate_shots(
                self.Snowflurry.sf_circuit, shots
            )
            if shots_results is None:
                logger.error("Simulation failed, please check your circuit.")
                raise measurement_error.MeasurementError("Simulation failed, please check your circuit.")
            result = dict(Counter(shots_results))
            return result
        else:  # if we have a client, we use the real machine
            converter.apply_readouts(mp.obs)
            qpu = self.Snowflurry.AnyonYamaskaQPU(
                self.Snowflurry.currentClient, self.Snowflurry.seval("project_id")
            )
            if not qpu.is_connected():
                logger.error("QPU is not connected, please check your connection.")
                raise measurement_error.MeasurementError("QPU is not connected, please check your connection.")
            shots_results, time = self.Snowflurry.transpile_and_run_job(
                qpu, self.Snowflurry.sf_circuit, shots
            )
            if shots_results is None:
                logger.error("Job failed, please check your circuit.")
                raise measurement_error.MeasurementError("Job failed, please check your circuit.")
            result = dict(Counter(shots_results))
            return result
