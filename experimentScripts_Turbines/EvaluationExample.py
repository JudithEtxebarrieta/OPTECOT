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

import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt
import time
#==================================================================================================
# CARGA DE DATOS NUMÉRICOS-EXPERIMENTALES (BASE DE DATOS POLAR DEL ARCHIVO 'POLARS.XLSX')
#==================================================================================================

# Los siguientes dos fragmentos de código están destinados para mostrar la diferencia entre cargar una 
# base de datos desde un archivo de Excel y cargarla directamente desde una base de datos serializada 
# (previamente procesada previamente). En el último caso, la operación de carga es mucho más eficiente 
# (es decir, consume menos tiempo), ya que el código no requiere abrir un archivo de Excel, leer cada 
# entrada, volcarlo en una variable separada y realizar la extensión/inversión. Simplemente carga un 
# objeto (un archivo Pickle), que está en formato binario y se almacenó al preprocesar la base de datos 
# en una primera ejecución.

# El argumento de la función que determina si se realiza el preprocesado o, en su lugar, se lee el archivo
# serializado, es el parámetro 'pick'. De forma predeterminada, 'pick' se establece igual a False, por lo 
# que el primer fragmento de código realiza la carga y el preprocesamiento del archivo de Excel, mientras 
# que el segundo carga un archivo Pickle serializado en su lugar. La diferencia en los tiempos de cómputo 
# (400 ms vs 20 ms aproximadamente) muestra la importancia de contar con un objeto serializado para este 
# tipo de operaciones.

polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)

#==================================================================================================
# FUNCIÓN FITNESS: VERSIÓN BLACK-BOX
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# DEFINICIÓN DE ESPACIOS PARAMÉTRICOS PARA VARIABLES
#--------------------------------------------------------------------------------------------------

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

#--------------------------------------------------------------------------------------------------
# DEFINICIÓN DE GENES POSITIVOS Y UN SUSTITUTO PARA UN INDIVIDUO DE POBLACIÓN GENÉRICA
#--------------------------------------------------------------------------------------------------

# Blade-number gene.
blade_number = [3, 5, 7]

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

# El 'turb_params' pretende ser un sustituto del individuo de la población en un GA basado en variables continuas.
turb_params = [np.random.choice(blade_number),
               np.random.uniform(sigma_hub[0], sigma_hub[1]),
               np.random.uniform(sigma_tip[0], sigma_tip[1]),
               np.random.uniform(nu[0], nu[1]),
               np.random.uniform(tip_clearance[0], tip_clearance[1]),# Línea modificada por el error comentado en las líneas 82-92
               np.random.choice(airfoil_dist)]

#--------------------------------------------------------------------------------------------------
# DEFINICIÓN DE ARGUMENTOS CONSTANTES
#--------------------------------------------------------------------------------------------------

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

#--------------------------------------------------------------------------------------------------
# TURBINA INSTANTÁNEA
#--------------------------------------------------------------------------------------------------
os.chdir('OptimizationAlgorithms_KONFLOT')
turb = turbine_classes.instantiate_turbine(constargs, turb_params)
os.chdir('../')



#--------------------------------------------------------------------------------------------------
# LLAMADA A LA FUNCIÓN FITNESS (evaluación de la función objetivo)
#--------------------------------------------------------------------------------------------------
t=time.time()
scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')
print('TIEMPO DE EJECUCIÓN DE UNA EVALUACIÓN:'+str(time.time()-t))
for e, score in enumerate(scores[0]):
    print("Fitness at sea-state #" + str(e) + ": " + str(score) + "\n")
print("Total fitness: " + str(scores[1]))
