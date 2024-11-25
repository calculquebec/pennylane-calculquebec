import copy
import pennylane_snowflurry.utility.kmp as kmp
import random as rand

class TestKmp:
    def alter(pattern):
        result = copy.deepcopy(pattern)
        test = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        alterations = [lambda s, i : s[:i] + [rand.choice(test)]+ s[i:], 
                 lambda s, i : s[:i] + s[i+1:], 
                 lambda s, i : s[:i] + [rand.choice(test)] + s[i+1:]]
        alt_count = rand.randint(1, 7)
        for _ in range(alt_count):
            result = rand.choice(alterations)(result, rand.randint(0, len(result) - 1))
        return result

    def test_find_pattern(self):
        string = list("klfn sfelixljfn skjfn skjn skjfelixfn szlsidi JSeliaeg felix slgij rpgjfnaukgn felixksjdbf ksjdbfelixgf slisdj asldffelixjisf")
        pattern = list("felix")
        index = kmp.kmp_search(string, pattern, lambda a, b: a == b)
        answer = 6
        assert index == answer