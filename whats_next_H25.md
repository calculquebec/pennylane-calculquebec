## perspective pour la suite du projet

Le projet discuté est de permettre de faire rouler les circuits de plusieurs utilisateurs en parallèle sur la machine. 

Le projet de joindre plusieurs circuits d'utilisateurs semble a priori découplé du projet de transpilation fait durant le stage automne 2024. J'aurais tendance à approcher le projet d'un de ces axes-ci : 

### faire la jointure de circuit indépendament du transpiler

- une possibilité pourrait être de garder le projet de transpilateur comme il est fait présentement, et de poser l'étape de jointure de circuits du côté de l'API, chez Calcul Québec. Ainsi, le device chez le client aurait la responsabilité de faire les étapes de transpilation sur son propres circuit et le travail de jointure se ferait en aval, sur les multiples circuits transpilés. 

Une fois les multiples circuits transpilés fusionnés ensemble, une job est créée et envoyée à MonarQ. Cette tâche retourne un résultat pour tous les cicruits. Ce résultat est décomposé

Ce travail nécessiterait de travailler avec le code JSON formatté destiné à Thunderhead. Il pourrait être nécessaire ou non de transformer les circuits sous format JSON en QuantumTapes, pour pouvoir plus facilement les manipuler et les joindre ensemble. J'ai mis un exemple dans la documentation de à quoi ressemble un circuit sous la forme de dictionnaire en JSON.

Il est important de penser à changer les numéros de qubits pour éviter les collision, tout en prenant en compte la topologie de monarq pour éviter de choisir des fils dont les chemins de coupleurs se croisent. 

Avantages : 
* engendre moins de changements au niveau du code et de la structure du projet existant. 
* engendre moins de changements et de traitement sur les circuits une fois qu'ils sont envoyés
* plus de modifiabilité pour l'utilisateur

Désavantage : 
* plus de traitement côté utilisateur peut vouloir dire plus de latence dans l'exécution des circuits
* peut rendre la tâche de fusion / mapping avec la topologie de la machine plus complexe


### "joindre" le transpiler et le processus de jointure de circuit

- Une autre possibilité pourrait être d'envoyer les circuits non-transpilés à un service qui s'occuperait de la transpilation et du post-processing du côté serveur. Cela pourrait être fait par l'intermédiaire d'une API qui prendrait les circuits formattés en JSON avec les informations de fils et de readout, et les transformerait, du côté du serveur, en QuantumTape pour pouvoir les fusionner ensemble en changeant les numéros de fils pour éviter les collisions. Celà fait, les étapes de transpilation et de post-processing pourraient être faites sur le QuantumTape non-transpilé, fusionné, avant d'être envoyé sur MonarQ. 

Avantages : 
* Tout le heavy lifting (Fusion, transpilation, post-processing) se fait côté serveur, et donc potentiellement sur monarq
* Le processus de fusionner les circuits est potentiellement plus simple puisqu'il n'y a pas encore eu de placement

Désavantage : 
* moins de modifiabilité côté utilisateur
* Grosse restructuration du code (déplacer le code du transpiler dans un API)

### why not both?

- Une dernière possibilité pourrait être de faire une étape de transpilation et de postprocessing "light weight" du côté de l'utilisateur, et de faire une étape de transpilation et de post processing plus poussé du côté de l'API. Cela semble être l'option qui offre le plus d'avantage et de mitigation des inconvénients. Cela permet de réutiliser beaucoup du code existant tout en permettant une évolutivité. 