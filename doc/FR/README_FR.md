# pennylane-calculquebec

## Contenu

- [Définitions](#definitions)
- [Structure du projet](#structure-du-projet)
- [Installation locale](#installation-locale)
- [Utilisation](#utilisation)
    - [Lancer des fichiers](#lancer-des-fichiers)
- [Dépendances](#dependances)
    - [Julia](#julia)
    - [Modules Python](#modules-python)
- [État du projet et problèmes connus](#etat-du-projet-et-problemes-connus)
    - [Plans futurs](#plans-futurs)
- [Réferences](#references)


## Definitions

Pennylane-CalculQuebec est un plugin PennyLane qui permet de lancer des tâches de manière transparente sur MonarQ, l'ordinateur quantique sans but lucratif de Calcul Québec.

Le plugin offre aussi des fonctionnalités de simulation et de pré/post traitement relatifs à l'ordinateur quantique MonarQ. 

[Pennylane](https://pennylane.ai/) est une librairie Python multiplateforme pour l'apprentissage machine quantique, la différentiation automatique et l'optimisation de calculs hybrides quantique-classique.

[Calcul Quebec](https://www.calculquebec.ca/) est un organisme sans but lucratif qui regroupe les universités de la province de Québec et fournit de la puissance de calcul aux milieux académique et de la recherche.  

[Snowflurry](https://snowflurry.org/) est un cadriciel de calcul quantique développé en Julia par Anyon Systems qui a pour objectif de donner accès à du matériel et des simulateurs quantique.

## Structure du projet

Comme présenté dans le diagramme ci-dessous, ce plugin contient un [device](https://pennylane.ai/plugins/) PennyLane appelé `monarq.default`. Ce device est définit par une classe `MonarqDevice`. Le device applique tout d'abord au circuit une série d'étapes de pré-traitement pour le simplifier et le rendre exécutable sur MonarQ. Le device crée et soumet en suite une Job en utilisant des appels à une API, et récupère les résultats quand ils sont prêts. Une série d'étape de post-traitement est alors appliquée et le résultat traité est retourné à l'utilisateur. 

Un autre device appelé `snowflurry.qubit` cohabite dans ce progiciel. Ce dernier fonctionne en convertissant le circuit PennyLane en circuit Snowflurry grâce à des outils comme JuliaCall qui permet la communication entre les environnements Python et Julia. Le circuit Snowflurry peut alors être utilisé avec les services disponibles, soit un simulateur, soit un ordinateur quantique réel. Le résultat est alors retourné dans PennyLane et formatté pour le retour à l'utilisateur. 

![project_structure](https://raw.githubusercontent.com/calculquebec/pennylane-calculquebec/9276a260959c886eed87373b74090a9d652b130c/doc/assets/project_structure.png)

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

Le plugin s'occupera aussi d'installer Julia et les modules Julia demandés, tel que Snowflurry et PythonCall durant la première exécution du device `snowflurry.qubit`.

## Utilisation

Si vous avez besoin de plus d'information sur le plugin, vous pouvez lire la page de [prise en main](https://github.com/calculquebec/pennylane-calculquebec/blob/main/doc/FR/prise_en_main.ipynb).

### Executer des fichiers

Le plugin peut être utilisé à la fois dans des scripts Python ou dans l'environnement Jupyter Notebook. Pour exécuter un script, utilisez la commande suivante : 

```sh
python base_circuit.py
```

## Dependances

### Julia

À partir de la version 0.3.0, **il n'est plus nécessaire d'installer Julia manuellement** puisque le plugin se charge de télécharger et d'installer la version requise automatiquement à la première utilisation. L'environnement Julia est alors assigné au plugin. 

Par contre, si vous souhaiter gérer votre environnement, vous pouvez télécharger Julia à partir du [site web officiel](https://julialang.org/downloads/). Il est fortement recommandé de l'installer en utilisant le fichier d'installation, puisque cela vous permettra d'ajouter les variables d'environnement relatives à Julia. 

**Pour être certain que la configuration est correcte, durant le processus d'installation, la case à cocher `Add Julia to PATH` doit être cochée**

Depuis la version 0.5.0 **Julia sera seulement installé si si vous utilisez Snowflurry.Qubit**. Vous pouvez utiliser Monarq.Default sans dépendance à Julia. 

### Modules Python

Ces modules sont installés automatiquement durant le processus d'installation du plugin, et son nécessaire à son fonctionnement. Voici les liens ci-dessous :

- Pour PennyLane, veuillez vous référer à cette [documentation](https://pennylane.ai/install/).

- Pour Snowflurry, veuillez vous référer à cette [documentation](https://snowflurry.org).

- Netowkx est une librairie d'algortithmes de graphes en Python. Elle est utilisé de manière transparente au courant de certaines étapes de transpilation. Voici la [documentation](https://networkx.org/).

- Numpy est une librairie mathématique grandement utilisée par PennyLane et par le plugin. Voici la [documentation](https://numpy.org/doc/2.1/index.html).

## Etat du projet et problemes connus

Le plugin est présentement en phase béta et fournit un accès à MonarQ directement à travers des appels d'API. Il contient aussi des fonctionnalité permettant d'obtenir des métriques et des information sur la machine. Le plugin contient aussi des fonctionnalités permettant aux utilisateurs avancés de changer les étapes de pré/post traitement et de créer des étapes personnalisées. Le plugin contient un simulateur pouvant être accédé avec le nom `monarq.sim`, mais certains ajustements au niveau du modèle de bruit sont nécessaires pour mimiquer le plus fidèlement possible le modèle de bruit de MonarQ. Les étapes de placement et de routage permettent théoriquement de trouver des qubits et coupleurs de qualité en fonction des fidélités de ces derniers, mais le modèle de bruit n'étant pas encore complet, les résultats ne sont pas encore optimaux. Des étapes de post-traitement ont été ajoutées pour améliorer la fidélité de mesure. La couverture de test est présentement de plus de 90 % (14-04-2025). 

### Plans futurs

- Intégrer des fonctions de parallélisation de circuit pour exécuter plusieurs circuits simultanément
- Ajouter des nouvelles étapes de traitement au device pour améliorer le placement, le routage et l'optimisation

## References 

Le wiki de Calcul Québec fournit beaucoup d'information sur le plugin, ses composants et sur comment les utiliser. Vous pouvez y accéder [ici](https://docs.alliancecan.ca/wiki/Services_d%27informatique_quantique).
