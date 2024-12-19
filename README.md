# pennylane-calculquebec

## Table of content

- [Definitions](#definitions)
- [Project structure](#project-structure)
- [Local installation](#local-installation)
- [Usage](#usage)
    - [Running files](#running-files)
- [Dependencies](#dependencies)
    - [Julia](#julia)
    - [Python modules](#python-modules)
- [State of the project and known issues](#state-of-the-project-and-known-issues)
    - [Future plans](#future-plans)
- [References](#references)


## Definitions

Pennylane-CalculQuebec is a PennyLane plugin enabling seamless job execution on MonarQ, Calcul Québec’s nonprofit-hosted quantum computer. 

It also offers simulation and pre-processing / post-processing capabilities relative to MonarQ quantum computer.

[Pennylane](https://pennylane.ai/) is a cross-platform Python library for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

[Calcul Quebec](https://www.calculquebec.ca/) is a non-profit organization that regroups universities from the Province of Quebec and provides computing power to research and academia.  

[Snowflurry](https://snowflurry.org/) is a quantum computing framework developed in Julia by Anyon Systems and aims to provide access to quantum hardware and simulators.

## Project structure

As shown in the diagram below, this plugin contains a Pennylane [device](https://pennylane.ai/plugins/) named `monarq.default`. This devices is defined by the class `MonarqDevice`. It first applies a series of pre-processing steps to the circuit in order to simplify it, and make it executable on MonarQ. It then creates and submits a job using API calls, and gets the results back once they're ready. A series of post-processing steps are then applied, and the processed results are returned to the end user. 

An other device named `snowflurry.qubit` also lives in the package. It works by converting a PennyLane circuit into a Snowflurry circuit, thanks to packages like JuliaCall that allow the communication between Python and Julia environments. The The Snowflurry circuit can then be used with the available backends, either a simulator or real quantum hardware. The results are then converted back into PennyLane's format and returned to the user.

![project_structure](https://raw.githubusercontent.com/calculquebec/pennylane-calculquebec/9276a260959c886eed87373b74090a9d652b130c/doc/assets/project_structure.png)

## Local installation

Pennylane-calculquebec can be installed using pip:

```sh
pip install pennylane-calculquebec
```

Alternatively, you can clone this repo and install the plugin with the following command from the root of the repo:

```sh
pip install -e .
```

Pennylane and other Python dependencies will be installed automatically during the installation process.

The plugin will also take care of installing Julia and the required Julia packages, such as Snowflurry and PythonCall during the first run of the Snowflurry.Qubit device. 

## Usage

If you need more information about how to use the plugin, you may read the [getting-started](./doc/getting_started.ipynb) jupyter notebook.

### Running files

The plugin can be used both in python scripts and Jupyter notebooks. To run a script, you can use the following command:

```sh
python base_circuit.py
```

## Dependencies

### Julia

As of version 0.3.0, **there is no need to install Julia manually**, since the plugin will download and install the required version automatically upon first use. This Julia environment is bound to the plugin.

However, if you wish to manage your Julia environment, you can download it from the [official website](https://julialang.org/downloads/). It is highly recommended to install using the installer file, as it will ask to add Julia to the system's environment variables.

**To ensure this correct configuration, during the installation process, the checkbox `Add Julia to PATH` must be checked.**

As of version 0.5.0, **Julia will only be installed if you use Snowflurry.Qubit**. You can use Monarq.Default without depending on Julia. 

### Python modules

Those packages are installed automatically during the plugin installation process and are necessary for the plugin to work. Here are the links to their respective documentation:

- For PennyLane, please refer to the [PennyLane documentation](https://pennylane.ai/install/).

- For Snowflurry, please refer to the [Snowflurry documentation](https://snowflurry.org).

- Netowkx is a Python graph algorithms library. It is used seemlessly during some of the transpiling steps. Here's the [documentation](https://networkx.org/).

- Numpy is a mathematical library that is used heavily by Pennylane, and the plugin. Here's the [documentation](https://numpy.org/doc/2.1/index.html).

## State of the project and known issues

The plugin is currently in its beta phase and provides access to MonarQ directly through API calls. It also contains capabilities for obtaining benchmarks and machine informations. There are also features that let experimented users change the pre-processing / post-processing behaviour of the device, and create custom pre-processing / post processing steps. There is a simulator device which is currently being developed, but the noise model still needs to be tweaked. The placement and routing phases of the transpiler currently chose wires and couplers by prioritizing best fidelities first, but this does not yield optimal results in terms of errors. The unit-test coverage is still not complete.

### Future plans

- Have 80 % unit test line coverage for each file in the project
- Integrate circuit paralellization capabilities to run multiple circuits at the same time
- Add new transpiling steps to the device to improve placement, routing, error mitigation and optimization.
- Make the MonarQ simulation device available through qml.device

## References 

Calcul Québec's Wiki provides a lot of information on the plugin, its components and how to use them. You can access it [here](https://docs.alliancecan.ca/wiki/Les_services_quantiques).
