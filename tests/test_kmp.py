import copy
import pennylane_calculquebec.utility.kmp as kmp
import random as rand

class TestKmp:

    def test_find_pattern(self):
        string = list("klfn sfelixljfn skjfn skjn skjfelixfn szlsidi JSeliaeg felix slgij rpgjfnaukgn felixksjdbf ksjdbfelixgf slisdj asldffelixjisf")
        pattern = list("felix")
        index = kmp.kmp_search(string, pattern, lambda a, b: a == b)
        answer = 6
        assert index == answer