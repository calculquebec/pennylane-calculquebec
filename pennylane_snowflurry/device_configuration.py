import pennylane_snowflurry.transpiler.transpiler_enums as Enums

class Client:
    """
    data object that is used to pass client information to CalculQCDevice
    
    Properties : 
    host : the server address for the machine
    user : the users identifier
    access_token : the unique access key provided to the user
    realm : the organisational group associated with the machine
    machine_name : the name of the machine
    project_name : the name of the project
     
    """
    host : str
    user : str
    access_token : str
    realm : str
    machine_name : str
    project_name : str
    
    def __init__(self, host : str, user : str, access_token : str, realm : str, machine_name : str, project_name : str):
        self.host = host
        self.user = user
        self.access_token = access_token
        self.realm = realm
        self.machine_name = machine_name
        self.project_name = project_name

    
class CalculQuebecClient(Client):
    """
    specialization of Client for Calcul Quebec infrastructures
    
    Properties : 
    host : the server address for the machine
    user : the users identifier
    access_token : the unique access key provided to the user
    realm : the organisational group associated with the machine (calculqc)
    machine_name : the name of the machine
    project_name : the name of the project
     
    """
    def __init__(self, host, user, token, machine_name, project_name):
        super().__init__(host, user, token, "calculqc", machine_name, project_name)


class MonarqClicent(CalculQuebecClient):
    """
    specialization of CalculQuebecClient for MonarQ infrastructure
    
    Properties : 
    host : the server address for the machine
    user : the users identifier
    access_token : the unique access key provided to the user
    realm : the organisational group associated with the machine (calculqc)
    machine_name : the name of the machine (yamaska)
    project_name : the name of the project
     
    """
    def __init__(self, host, user, token, project_name = ""):
        super().__init__(host, user, token, "yamaska", project_name)


class Config:
    pass


class MonarqConfig(Config):
    baseDecomposition : Enums.BaseDecomp
    placement : Enums.Place
    routing : Enums.Route
    optimization : Enums.Optimization
    nativeDecomposition : Enums.NativeDecomp
    useBenchmark : Enums.Benchmark
    def __init__(self, baseDecomposition = Enums.BaseDecomp.CLIFFORDT, 
                 placement = Enums.Place.ASTAR, 
                 routing = Enums.Route.ASTARSWAP,
                 optimization = Enums.Optimization.COMMUTEANDMERGE,
                 nativeDecomposition = Enums.NativeDecomp.MONARQ, 
                 useBenchmark = Enums.Benchmark.ACCEPTANCE):
        self.baseDecomposition = baseDecomposition
        self.placement = placement
        self.routing = routing
        self.optimization = optimization
        self.nativeDecomposition = nativeDecomposition
        self.useBenchmark = useBenchmark