
#==================================================================================================
# LIBRERIAS
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

from cma.utilities import utils 
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

#==================================================================================================
# CLASES
#==================================================================================================

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

#==================================================================================================
# FUNCIONES
#==================================================================================================

def choose_random_configuration(seed,change_seed,blade_number):
	# Definir rangos de los parametros que definen el diseno de la turbina.
	sigma_hub = [0.4, 0.7]# Hub solidity gene.
	sigma_tip = [0.4, 0.7]# Tip solidity gene.
	nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
	tip_clearance=[0,3]# Tip-clearance gene.	  
	airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

	# Obtener configuracion aleatoria.
	if change_seed:
		np.random.seed(seed)

	turb_params = [blade_number,
	       np.random.uniform(sigma_hub[0], sigma_hub[1]),
	       np.random.uniform(sigma_tip[0], sigma_tip[1]),
	       np.random.uniform(nu[0], nu[1]),
	       np.random.uniform(tip_clearance[0], tip_clearance[1]),
	       np.random.choice(airfoil_dist)]


	return turb_params

def build_constargs_dict(N):
	# Definir parametros constantes.
	omega = 2100# Rotational speed.
	rcas = 0.4# Casing radius.
	airfoils = ["NACA0015", "NACA0018", "NACA0021"]# Set of possible airfoils.
	polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)# Polars.
	cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]# Target dumping coefficients.
	devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]# Input sea-state standard pressure deviations.
	weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]# Input sea-state weights.
	Nmin = 1000#Max threshold rotational speeds
	Nmax = 3200#Min threshold rotational speeds

	# Construir el diccionario que necesita la funcion fitness
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

def evaluation(turb_params,N,count_time):

	# Construir diccionario de parametros constantes.
	constargs=build_constargs_dict(N)

	# Crear turbina instantantanea.
	os.chdir('OptimizationAlgorithms_KONFLOT')
	turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
	os.chdir('../')

	# Calcular evaluacion.
	if count_time:
		sw.resume()
	scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')
	if count_time:
		sw.pause()

	return scores[1]

def learn(seed,default_N,N,blade_number,max_time):

	change_seed=True

	list_turb_params=[]
	list_scores=[]

	while sw.get_time()<max_time:
		
		# Escoger un nuevo diseno de turbina.
		turb_params=choose_random_configuration(seed,change_seed,blade_number)
		change_seed=False
		list_turb_params.append(turb_params)

		# Evaluar el nuevo diseno de turbina.
		turb_score=evaluation(turb_params,N,True)
		list_scores.append(turb_score)

		# Evaluar mejor diseno de turbina seleccionado hasta el momento.
		# best_turb_params=list_turb_params[list_scores.index(max(list_scores))]
		# eval_score=evaluation(best_turb_params,default_N,False)

		# Actualizar base de datos.
		# df_acc.append([N,seed,len(list_turb_params),turb_score,eval_score,sw.get_time()])

	return list_turb_params

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

#--------------------------------------------------------------------------------------------------
# Ejecucion normal.
#--------------------------------------------------------------------------------------------------

# Valor por defecto del parametro.
default_N=50

# Limite de tiempo de entrenamiento.
max_time=60*9

# Blade-number.
blade_number=3

# Mallados.
list_seeds=range(0,10,1)
list_acc=[1.0,0.8,0.6,0.4,0.2]

# Por cada valor de accuracy guardar una base de datos.
for accuracy in list_acc:
	# Calcular el valor de N correspondiente.
	N=int(default_N*accuracy)

	# Cosntruir base de datos.
	df_acc=[]
	for seed in tqdm(list_seeds):

		# Inicializar contador de tiempo.
		sw=stopwatch()
		sw.pause()

		# Entrenamiento.
		list_turb_params=learn(seed,default_N,N,blade_number,max_time)

		# Cuando estemos con el accuracy minimo (con el que mas turbinas se podran evaluar), se guardaran los disenos de turbinas evaluados.
		if accuracy==min(list_acc):
			df_turb_params=pd.DataFrame(list_turb_params)
			df_turb_params.to_csv('results/data/Turbines/ConstantAccuracyAnalysis_RandomPopulation/df_turb_params_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv')

	# Guardar base de datos asociada al accuracy fijado.
	df_acc=pd.DataFrame(df_acc,columns=['N','seed','n_turb','turb_score','eval_score','elapsed_time'])
	df_acc.to_csv('results/data/Turbines/ConstantAccuracyAnalysis_RandomPopulation/df_blade_number'+str(blade_number)+'_train_acc'+str(accuracy)+'.csv')

# Guardar lista de accuracys.
np.save('results/data/Turbines/ConstantAccuracyAnalysis_RandomPopulation/list_acc',list_acc)

# Guardar limite de tiempo de entrenamiento.
np.save('results/data/Turbines/ConstantAccuracyAnalysis_RandomPopulation/max_time',max_time)

