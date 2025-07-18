from .measurement_strategy import MeasurementStrategy
import numpy as np
from calcul_quebec_error import measurement_error
from pennylane_calculquebec import logger

class Sample(MeasurementStrategy):

    def __init__(self):
        super().__init__()

    def measure(self, converter, mp, shots):
        if self.Snowflurry.currentClient is None:
            converter.remove_readouts()
            converter.apply_readouts(mp.obs)
            shots_results = self.Snowflurry.simulate_shots(
                self.Snowflurry.sf_circuit, shots
            )
            if not shots_results:
                logger.warning(
                    "No shots were returned. This may be due to an empty circuit or an error in the simulation."
                )
                raise measurement_error.MeasurementError("No shots returned from simulation.")
            return np.asarray(shots_results).astype(int)
        else:
            converter.apply_readouts(mp.obs)
            qpu = self.Snowflurry.AnyonYamaskaQPU(
                self.Snowflurry.currentClient, self.Snowflurry.seval("project_id")
            )
            if qpu is None:
                logger.warning(
                    "No QPU available. Please ensure you have a valid QPU connection."
                )
                raise measurement_error.MeasurementError("No QPU available for measurement.")
            shots_results, time = self.Snowflurry.transpile_and_run_job(
                qpu,
                self.Snowflurry.sf_circuit,
                shots,
            )
            if not shots_results:
                logger.warning(
                    "No shots were returned from the QPU. This may indicate an issue with the QPU or the circuit."
                )
                raise measurement_error.MeasurementError("No shots returned from QPU.")
            return np.repeat(
                [int(key) for key in shots_results.keys()],
                [value for value in shots_results.values()],
            )
