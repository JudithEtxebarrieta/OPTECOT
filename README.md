# OPTECOT - _Optimal Evaluation Cost Tracking_ to maximice optimization problem solution quality in a given runtime 

## Repository contents

This repository contains the supplementary material for the paper _Optimal evaluation cost tracking to maximice optimization problem solution quality in a given runtime_. In this work we have presented a technique to reduce the cost of solving a computationally expensive black-box optimization problem using population-based algorithms, avoiding loss of solution quality. For this purpose, we have defined a optimal evaluation cost tracking heuristic (OPTECOT) capable of selecting the optimal evaluation cost during the algorithm execution process. The effectiveness of the proposal has been demonstrated in four different environments: **Symbolic Regressor**, **WindFLO**, **Swimmer** (from MuJoCo) and **Turbines**. In addition, future work has been motivated by using the environment **CartPole**. 

### Experiment Scripts
The scritp associated with the experiments performed in each environment are located in the following folders: **experimentScripts**_<font color="grey">**Environment**</font>. The experiments shown in the paper correspond to Python files with names starting with: **UnderstandingAccuracy**, **ConstantAccuracyAnalysis** and **OptimalAccuracyAnalysis**. The rest are additional experiments that have not been named in the paper.

### Results
The results of the experiments are distributed in two different folders:

- **results.** In this folder you can find both numerical and graphical results of the above mentioned experiments.
- **figures_paper.** In this folder you can find the adapted versions of some of the above mentioned scripts with the ending **_figures.py** in their name. The scripts in this folder adapt the code of the original files to build the graphs that are introduced in the paper.

### Install dependencies
To run the scripts from this repository, you must install the dependencies indicated in the file: **setup.sh**.

<font color="red">
Comprobar que este archivo contiene la información necearía para el funcionamiento del código.
</font>

```
bash others/setup.sh
```

## Brief description and results
<figure class="image">
<img src="others/readme_images/diagram_proposal_text.
png" width="100%"> 
<figcaption> Figure 1. Summary of the problem definition and the proposed procedure for its resolution.</figcaption>
</figure>

### What problems does OPTECOT solve?
<font color="red">
El heuristics esta diseñado para ser aplicado sobre algoritmos de optimización basados en poblaciones. Ademas, para poder ser aplicado se debe disponer de un parámetro que forma parte de la función objetivo y cuya modificación nos permite controlar el coste computacional de la función objetivo.

Aunque OPTECOT solo haya sido aplicado sobre cuatro entornos en este trabajo, es aplicable sobre cualquier otro entorno que cumpla los requisitos explicados arriba.
</font>

### How OPTECOT works?
<font color="red">
OPTECOT esta diseñado para reducir el coste computacional de los algoritmos basados en poblaciones cuando el coste por evaluación de la función objetivo es muy elevado. En este contexto en el que una evaluación es costosa, este tipo de algoritmos nos proporcionan una solución cercana a la optima siempre y cuando estemos dispuestos a aceptar un coste elevado. Para resolver el problema, OPTECOT selecciona el coste de evaluación menor (asociado a un valor menos preciso del parametro) que nos sigue proporcionando una solución de calidad (correcta clasificación de una población) durante el proceso de ejecución del algoritmo.

Nota.- intentar meter notación de la Figura 1 para que se pueda entender el diagrama pero sin pasarse de dificultad.
</font>

### Is OPTECOT effective?
<font color="red">
La efectividad de OPTECOT se a demostrado sobre 4 entornos muy diferentes entre si.
</font>

#### Experimental results
<font color="red">
Meter figura de aplicación de heuristico junto a curvas del comportamiento del coste.
</font>

#### Relation between cost (used in the paper) and accuracy (used in the code)

<font color="red">
Hacer una tabla con los valores por defecto del parametro seleccionados para cada entorno. 

Definir relación entre accuracy y coste.
</font>









