"""
Contains base client class and implementations. \n
MonarQ users will mostly use MonarqClient.
"""

class ApiClient:
    """
    data object that is used to pass client information to CalculQCDevice
    
    Args : 
        host (str) : the server address for the machine
        user (str) : the users identifier
        access_token (str) : the unique access key provided to the user
        realm (str) : the organisational group associated with the machine
        machine_name (str) : the name of the machine
        project_name (str) : the name of the project
     
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

    
class CalculQuebecClient(ApiClient):
    """
    specialization of Client for Calcul Quebec infrastructures
    
    Args : 
        host (str) : the server address for the machine
        user (str) : the users identifier
        access_token (str) : the unique access key provided to the user
        machine_name (str) : the name of the machine
        project_name (str) : the name of the project
     
    """
    def __init__(self, host, user, token, machine_name, project_name):
        super().__init__(host, user, token, "calculqc", machine_name, project_name)


class MonarqClient(CalculQuebecClient):
    """
    specialization of CalculQuebecClient for MonarQ infrastructure
    
    Args : 
        host (str) : the server address for the machine
        user (str) : the users identifier
        access_token (str) : the unique access key provided to the user
        project_name (str) : the name of the project
     
    """
    def __init__(self, host, user, access_token, project_name = ""):
        super().__init__(host, user, access_token, "yamaska", project_name)

