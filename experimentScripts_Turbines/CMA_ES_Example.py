# El código de este script está proporcionado por MGEP. En el se calcula la evaluación de un
# diseño de turbina aleatorio.

#==================================================================================================
# IMPORTAR PAQUETES
#==================================================================================================
import numpy as np
import matplotlib as mpl
import scipy as sc
import matplotlib.pyplot as plt
import scipy.integrate as scint
import time
import os
from scipy.interpolate import interp1d
from scipy.interpolate import LSQUnivariateSpline as lsqus
from scipy.integrate import simpson
from operator import itemgetter
import openpyxl
import pickle
import copy
from functools import partial
from scipy.interpolate import interp1d
from typing import Union
import cma
import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt
import time



# pip install git+https://github.com/CMA-ES/pycma.git@master



class stopwatch:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t


polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)


# Solidities.
sigmamin = 0.4
sigmamax = 0.7

# Hub-to-tip ratio.
hub_to_tip_min = 0.4
hub_to_tip_max = 0.75

# Set of possible airfoils.
airfoils = ["NACA0015", "NACA0018", "NACA0021"]

# Tip-clearance.
tip_clearance_min = 0
tip_clearance_max = 3

# Blade-number gene.
blade_numbers = [3, 5, 7]

# Hub solidity gene.
sigma_hub = [sigmamin, sigmamax]

# Tip solidity gene.
sigma_tip = [sigmamin, sigmamax]

# Hub-to-tip-ratio gene.
nu = [hub_to_tip_min, hub_to_tip_max]          

# Tip-clearance gene.
tip_clearance = [tip_clearance_min, tip_clearance_max]

# Airfoil dist. gene.    
airfoil_dist = np.arange(0, 27)

# Discretization: number of blade elements.
N = 50                                #<<<<<< ESTE ES EL PARAMETRO EQUIVALENTE AL time-step

# Rotational speed.
omega = 2100

# Casing radius.
rcas = 0.4

# Polars.
polars = polars

# Target dumping coefficients.
cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]

# Input sea-state standard pressure deviations and weights.
devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]
weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]

# Threshold rotational speeds (fitness value = 0 if the computed rotational speed falls outside the specified range).
Nmin = 1000
Nmax = 3200

# La función fintess requiere el diccionario 'constargs' para instanciar el objeto turbina y calcular la puntuación.
constargs = {"N": N,
             "omega": omega,
             "rcas": rcas,
             "airfoils": airfoils,
             "polars": polars,
             "cpobjs": cpobjs,
             "devobjs": devobjs,
             "weights": weights,
             "Nmin": Nmin,
             "Nmax": Nmax,
             "Mode": "mono"}







def fitness_funcion(turb_params, N):
    os.chdir('OptimizationAlgorithms_KONFLOT')
    constargs["N"] = N
    turb = turbine_classes.instantiate_turbine(constargs, turb_params)
    os.chdir('../')
    scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')
    return -scores[1] # We need to invert the fitness function because CMA-ES expects a minimization problem












# Optimizar con CMA-ES

bounds=np.array(
[
[sigma_hub[0]    , sigma_hub[1]],
[sigma_tip[0]    , sigma_tip[1]],
[nu[0]           , nu[1]],
[tip_clearance[0], tip_clearance[1]],
[0               , 26]
]
)


# Ejecutamos el algoritmo con x en el intervalo [0,1], por eso necesitamos escalar las variables para usar la funcion objetivo.
def scale_x(x):
    return x * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

# Preparar x para la funcion objectivo
def transformar_turb_params(x, blade_number):
    scaled_x = scale_x(x)
    return [blade_number]+list(scaled_x[:-1])+[round(scaled_x[-1])]



seed=2
blade_number = 3 # repetir el experimento con blade_number = 5 y blade_number = 7
N = 6
maxfevals=500 # max evals hay que decidirlo

# CMA-ES minimiza f(x), donde x es vector de tamaño 5 definido en [0,1]. Por eso hay que convertir x en los parametros de turbina con transformar_turb_params()
# Ademas, la funcion objetivo la he multiplicado por -1 porque necesitamos un problema de minimizacion.
np.random.seed(seed)
es = cma.CMAEvolutionStrategy(np.random.random(5), 0.33,inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfevals})
sw = stopwatch()
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [fitness_funcion(transformar_turb_params(x, blade_number), N) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    print("---------")
    print("Funcion objetivo: ", es.result.fbest)
    print("Mejor turbina so far: ", transformar_turb_params(es.result.xbest, blade_number))
    print("Evaluaciones funcion objetivo: ", es.result.evaluations)
    print("Tiempo: ", sw.get_time())
    print("---------")
    es.disp()
es.result_pretty()
cma.plot()  # shortcut for es.logger.plot()


