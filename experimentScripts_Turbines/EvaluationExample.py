'''This code is used to calculate the evaluation of a random turbine design.'''

#==================================================================================================
# LIBRARIES
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
# MAIN PROGRAM
#==================================================================================================

#--------------------------------------------------------------------------------------------------
# Define a random turbine design.
#--------------------------------------------------------------------------------------------------
# Definition of parameters to be optimized.
blade_number = [3, 5, 7] # Blade-number gene.
sigma_hub = [0.4, 0.7]# Hub solidity gene.
sigma_tip = [0.4, 0.7]# Tip solidity gene.
nu = [0.4, 0.75]# Hub-to-tip-ratio gene.       
tip_clearance = [0, 3]# Tip-clearance gene. 
airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.   


# Obtain a random turbine design.
turb_params = [np.random.choice(blade_number),
               np.random.uniform(sigma_hub[0], sigma_hub[1]),
               np.random.uniform(sigma_tip[0], sigma_tip[1]),
               np.random.uniform(nu[0], nu[1]),
               np.random.uniform(tip_clearance[0], tip_clearance[1]),
               np.random.choice(airfoil_dist)]

#--------------------------------------------------------------------------------------------------
# Definition of constant parameters.
#--------------------------------------------------------------------------------------------------
# Parameters.
N = 50  # Number of blade elements.                                          <<<<<<<<<<<<<<<<<<<<<< PARAMETER THAT CAN BE MODIFIED
airfoils = ["NACA0015", "NACA0018", "NACA0021"]# Set of possible airfoils.
omega = 2100# Rotational speed.
rcas = 0.4# Casing radius.
polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)# Database.
cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]# Target dumping coefficients.
devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]# Input sea-state standard pressure deviations.
weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]# Input sea-state weights.
Nmin = 1000# Min threshold rotational speeds.
Nmax = 3200# Max threshold rotational speeds.

# Dictionary with constant parameters (necessary to calculate an evaluation).
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
# Evaluate the selected turbine design.
#--------------------------------------------------------------------------------------------------
# Create instantaneous turbine.
os.chdir('OptimizationAlgorithms_KONFLOT')# Change default directory to be able to access COORDS.
turb = turbine_classes.instantiate_turbine(constargs, turb_params)
os.chdir('../')# Return to default directory.

# Calculate score and execution time.
t=time.time()
scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')
print('Evaluation time: '+str(time.time()-t))
for e, score in enumerate(scores[0]):
    print("Fitness at sea-state #" + str(e) + ": " + str(score) + "\n")
print("Total fitness: " + str(scores[1]))
