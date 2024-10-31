from enum import Enum

class Place(Enum):
    NONE = 0
    ASTAR = 1 << 0
    ISMAGS = 1 << 1
    VF2 = 1 << 2
    
class Route(Enum):
    NONE = 0
    ASTARSWAP = 1 << 0

class BaseDecomp(Enum):
    NONE = 0
    CLIFFORDT = 1 << 0

class NativeDecomp(Enum):
    NONE = 0
    MONARQ = 1 << 0

class Optimization(Enum):
    NONE = 0
    COMMUTEANDMERGE = 1 << 0
    
class Benchmark(Enum):
    NONE = 0
    ACCEPTANCE = 1 << 0