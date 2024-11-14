from pennylane_snowflurry.measurements.monarq_device.counts import Counts
from random import shuffle
class Sample:
    def measure(self, tape):
        result = Counts().measure(tape)
        
        samples = []
        
        for key in result:
            count = result[key]
            for _ in range(count):
                samples.append([int(i) for i in key])
        shuffle(samples)
        return samples
        