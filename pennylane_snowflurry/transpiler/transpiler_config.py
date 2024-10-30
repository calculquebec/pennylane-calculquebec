class TranspilerConfig:
    class Place:
        @property
        def NONE(): 0
        @property
        def ASTAR(): 1 << 0
        @property
        def ISMAGS(): 1 << 1
        @property
        def VF2(): 1 << 2
    
    class Route:
        @property
        def NONE(): 0
        @property
        def ASTARSWAP(): 1 << 0
    
    class BaseDecomp:
        @property
        def NONE(): 0
        @property
        def CLIFFORDT(): 1 << 0
    
    class NativeDecomp:
        @property
        def NONE(): 0
        @property
        def MONARQ(): 1 << 0
    
    class Optimization:
        @property
        def NONE(): 0
        @property
        def NAIVE(): 1 << 0
        
    class Benchmark:
        @property
        def NONE(): 0
        @property
        def ACCEPTANCE(): 1 << 0