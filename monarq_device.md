# Monarq Device

This document is aimed to explain what you need to know in order to use the MonarqDevice.

This device lets you communicate directly with MonarQ without using Snowflurry as an intermediate. This has the advantage of loosing the need to use Julia or other time consuming precompilings and calls. The MonarqDevice is also equipped with a compiler made by Calcul Quebec, which is optimized to work on MonarQ.

## default usage

1. install the plugin :

   ```
   pip install pennylane_snowflurry
   ```

2. import the dependencies :

   ```python
   from pennylane_snowflurry.device_configuration import Client
   ```

3. create a device

   ```python
   dev = qml.device("monarq.default", client=MonarqClient("host", "user", "access token", "realm"), wires=[0, 1, 2], shots=1000)
   ```

4. create your circuit

   ```python
   @qml.qnode(dev)
   def circuit():
       qml.Hadamard(0)
       qml.CNOT([0, 1])
       qml.CNOT([1, 2])
   ```

5. run your circuit and use results as you see fit

   ```python
   results = circuit()
   print(results)
   ```

## Client

In order to run jobs on Monarq, you will need to provide information to the device :

- host : the url to which monarq can be communicated with.
- user : your identifier
- access token : the access pass that identifies you as a rightful user
- realm : usually "calculqc"

in order to provide those informations to the device, you will need to store them in a ```Client```. This is done by :

- importing the client class :

    ```python
    from pennylane_snowflurry.device_configuration import MonarqClient
    ```

- creating a client and suplying your informations to it :

    ```python
    client = MonarqClient("host", "user", "access_token", "realm")
    ```

- passing the client to the device :

    ```python
    dev = qml.device("monarq.device", client=client, ...)
    ```

## configuration

The default behaviour of MonarqDevice's transpiler goes as such :

1. Decompose 3+ qubits gates and non-standard gates to a subset that is easier to optimize (clifford-t gates)
2. Map the circuit's wires to the machines qubits by using the circuit's and machine's connectivity graphs
3. If any 2+ qubit gates are not connected in given mapping, connect them using swaps
4. Optimize the circuit (commute gates, merge rotations, remove inverses and trivial gates)
5. Decompose non-native gates to native ones

The transpiler uses benchmarks of the machine's fiability in the placement and routing part.

You can change the behaviour of the transpiler by passing a MonarqConfig object to the device. This should only be done if you know what your doing because it could lead to your circuit not being ran.

Here's how to do it :

1. import the config and enums classes :

   ```python
   from pennylane_snowflurry.device_configuration import MonarqConfig
   import pennylane_snowflurry.transpiler.transpiler_enums as enums
   ```

2. create a config object and set the behaviour you want

   ```python
   # Disables placement, routing and benchmarking features, leaving decompositions and optimization unchanged from the base configuration
   config = MonarqConfig(placement=enums.Place.NONE, routing=enums.Route.NONE, useBenchmarking=enums.Benchmark.NONE) 
   ```

3. pass the config object to the device, along with the client and other arguments

   ```python
   dev = qml.device("monarq.qubit", client=client, config=config, ...)
   ```
