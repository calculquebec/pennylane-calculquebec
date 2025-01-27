from pennylane_calculquebec.processing.interfaces import PreProcStep, PostProcStep

class PrintTape(PreProcStep):
    def execute(self, tape):
        print(*tape.operations)
        return tape

class PrintResults(PostProcStep):
    def execute(self, tape, results):
        print(results)
        return results
    
class PrintWires(PreProcStep):
    def execute(self, tape):
        print(*[w for w in tape.wires])
        return tape