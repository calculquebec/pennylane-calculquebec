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
    _host : str
    _user : str
    _access_token : str
    _realm : str
    _machine_name : str
    _project_name : str
    
    def __init__(self, host : str, user : str, access_token : str, realm : str, machine_name : str, project_name : str):
        self._host = host
        self._user = user
        self._access_token = access_token
        self._realm = realm
        self._machine_name = machine_name
        self._project_name = project_name

    
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

class Behaviour:
    pass

class MonarqBehaviour:
    pass