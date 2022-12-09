# Mediante este script se guardan las imágenes del diseño de diferentes turbinas.

#==================================================================================================
# LIBRERÍAS
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

#==================================================================================================
# FUNCIONES
#==================================================================================================
def build_constargs_dict(N):

	# Definir parámetros constantes.
	omega = 2100# Rotational speed.
	rcas = 0.4# Casing radius.
	airfoils = ["NACA0015", "NACA0018", "NACA0021"]# Set of possible airfoils.
	polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)# Polars.
	cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]# Target dumping coefficients.
	devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]# Input sea-state standard pressure deviations.
	weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]# Input sea-state weights.
	Nmin = 1000#Max threshold rotational speeds
	Nmax = 3200#Min threshold rotational speeds

	# Construir el diccionario que necesita la función fitness.
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

	return constargs

def build_turbine(turb_params,N=50):
	# Construir diccionario de parámetros constantes.
	constargs=build_constargs_dict(N)

	# Crear turbina instantánea.
	os.chdir('OptimizationAlgorithms_KONFLOT')
	turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
	os.chdir('../')
	
	return turb

#==================================================================================================
# FUNCIONES
#==================================================================================================

# Parámetros de la turbina.
blade_number = [3, 5, 7]# Blade-number gene.
sigma_hub = [0.4, 0.7]# Hub solidity gene.
sigma_tip = [0.4, 0.7]# Tip solidity gene.
nu = [0.4, 0.75]  # Hub-to-tip-ratio gene.
tip_clearance=[0,3]# Tip-clearance gene.
airfoil_dist = np.arange(0, 27)# Airfoil dist. gene. 

# Mis ejemplos de turbinas.
turb1=build_turbine([3,0.4,0.4,0.4,0,0])
turb2=build_turbine([5,0.55,0.55,0.55,0.5,13.5])
turb3=build_turbine([7,0.7,0.7,0.7,1,13.5])

# Imagenes de las turbinas definidas.
fig,ax=turbine_classes.plot_turbine(turb1)
fig.savefig('results/figures/Turbines/ExampleTurb1.png')

fig,ax=turbine_classes.plot_turbine(turb2)
fig.savefig('results/figures/Turbines/ExampleTurb2.png')

fig,ax=turbine_classes.plot_turbine(turb3)
fig.savefig('results/figures/Turbines/ExampleTurb3.png')