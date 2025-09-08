from pennylane_calculquebec.processing.interfaces import PreProcStep, PostProcStep


class PrintTape(PreProcStep):
    """
    Print a tape as a preprocessing step
    """

    def execute(self, tape):
        """Shows the tape using the builtin print function without altering it

        Args:
            tape (QuantumTape): tape to print

        Returns:
            QuantumTape: unaltered tape
        """
        print(*tape.operations)
        return tape


class PrintResults(PostProcStep):
    """
    Prints the results of a tape
    """

    def execute(self, tape, results):
        """Prints the results of a tape without altering them

        Args:
            tape (QuantumTape): the circuit's quantum tape (unused)
            results (dict[str, int]): the circuit's results represented as counts

        Returns:
            dict[str, int]: the unaltered results
        """
        print(results)
        return results


class PrintWires(PreProcStep):
    """
    prints wires as a processing step using the print builtin python function
    """

    def execute(self, tape):
        """prints the wires a circuit acts upon without altering the circuit

        Args:
            tape (QuantumTape): The tape to print

        Returns:
            QuantumTape: the unaltered tape
        """
        print(*[w for w in tape.wires])
        return tape
