# pennylane-calculquebec

## Contenu

- [pennylane-calculquebec](#pennylane-calculquebec)
  - [Contenu](#contenu)
  - [Definitions](#definitions)
  - [Installation locale](#installation-locale)
  - [Utilisation](#utilisation)
    - [Executer des fichiers](#executer-des-fichiers)
  - [Dependances](#dependances)
    - [Modules Python](#modules-python)
  - [Etat du projet et problemes connus](#etat-du-projet-et-problemes-connus)
    - [Plans futurs](#plans-futurs)
  - [References](#references)


## Definitions

Pennylane-CalculQuebec est un plugin PennyLane qui permet de lancer des tâches de manière transparente sur MonarQ, l'ordinateur quantique sans but lucratif de Calcul Québec.

Le plugin offre aussi des fonctionnalités de simulation et de pré/post traitement relatifs à l'ordinateur quantique MonarQ. 

[Pennylane](https://pennylane.ai/) est une librairie Python multiplateforme pour l'apprentissage machine quantique, la différentiation automatique et l'optimisation de calculs hybrides quantique-classique.

[Calcul Quebec](https://www.calculquebec.ca/) est un organisme sans but lucratif qui regroupe les universités de la province de Québec et fournit de la puissance de calcul aux milieux académique et de la recherche.  

## Installation locale

Pennylane-calculquebec peut être installé en utilisant pip:

```sh
pip install pennylane-calculquebec
```

Alternativement, vous pouvez clôner ce répertoire git et installer le plugin avec cette commande à partir de la racine du répertoire : 

```sh
pip install -e .
```

Pennylane ainsi que toutes les dépendances Python seront installées automatiquement durant le processus d'installation.


## Utilisation

Si vous avez besoin de plus d'information sur le plugin, vous pouvez lire la page de [prise en main](https://github.com/calculquebec/pennylane-calculquebec/blob/main/doc/FR/prise_en_main.ipynb).

### Executer des fichiers

Le plugin peut être utilisé à la fois dans des scripts Python ou dans l'environnement Jupyter Notebook. Pour exécuter un script, utilisez la commande suivante : 

```sh
python base_circuit.py
```

## Dependances

### Modules Python

Ces modules sont installés automatiquement durant le processus d'installation du plugin, et son nécessaire à son fonctionnement. Voici les liens ci-dessous :

- Pour PennyLane, veuillez vous référer à cette [documentation](https://pennylane.ai/install/).


- Netowkx est une librairie d'algortithmes de graphes en Python. Elle est utilisé de manière transparente au courant de certaines étapes de transpilation. Voici la [documentation](https://networkx.org/).

- Numpy est une librairie mathématique grandement utilisée par PennyLane et par le plugin. Voici la [documentation](https://numpy.org/doc/2.1/index.html).

## Etat du projet et problemes connus

Le plugin est présentement en phase béta et fournit un accès à MonarQ directement à travers des appels d'API. Il contient aussi des fonctionnalités permettant d'obtenir des métriques et des informations sur la machine. Le plugin contient aussi des fonctionnalités permettant aux utilisateurs avancés de changer les étapes de pré/post traitement et de créer des étapes personnalisées. Le plugin contient un simulateur pouvant être accédé avec le nom `monarq.sim`, mais certains ajustements au niveau du modèle de bruit sont nécessaires pour mimiquer le plus fidèlement possible le modèle de bruit de MonarQ. Les étapes de placement et de routage permettent théoriquement de trouver des qubits et coupleurs de qualité en fonction des fidélités de ces derniers, mais le modèle de bruit n'étant pas encore complet, les résultats ne sont pas encore optimaux. Des étapes de post-traitement ont été ajoutées pour améliorer la fidélité de mesure. La couverture de test est présentement de plus de 90 % (14-04-2025). 

### Plans futurs

- Intégrer des fonctions de parallélisation de circuit pour exécuter plusieurs circuits simultanément
- Ajouter des nouvelles étapes de traitement au device pour améliorer le placement, le routage et l'optimisation

## References 

Le wiki de Calcul Québec fournit beaucoup d'information sur le plugin, ses composants et sur comment les utiliser. Vous pouvez y accéder [ici](https://docs.alliancecan.ca/wiki/Services_d%27informatique_quantique).
