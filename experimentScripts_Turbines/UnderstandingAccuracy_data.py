# Mediante este script se guarda en una base de datos la información relevante asociada a la 
# evaluación de un conjunto de configuraciones/diseños de turbinas. El conjunto esta formado por
# un total de 10 o 100 diseños de turbina escogidos de forma aleatoria, y todas las configuraciones del
# mismo se evalúan considerando diferentes valores de N. Se construirán unas bases de datos con la 
# información de scores y tiempos de ejecución por evaluación.

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

from tqdm import tqdm
import pandas as pd

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCIÓN 1
# Parámetros:
#   >list_seeds: lista de semillas a partir de las cuales se seleccionarán el
#    conjunto de turbinas aleatorias, (tantas turbinas como cardinal de esta lista).
# Devuelve: lista con configuraciones aleatorias de los parámetros que definen el diseño 
# de las turbinas.

def choose_random_configurations(list_seeds):
	# Definir rangos de los parámetros que definen el diseño de la turbina.
	blade_number = [3, 5, 7]# Blade-number gene.
	sigma_hub = [0.4, 0.7]# Hub solidity gene.
	sigma_tip = [0.4, 0.7]# Tip solidity gene.
	nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
	tip_clearance=[0,3]# Tip-clearance gene.	  
	airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

	# Guardar tantas configuraciones aleatorias como semillas en list_seeds.
	list_turb_params=[]
	for seed in list_seeds: 
		np.random.seed(seed)
		turb_params = [np.random.choice(blade_number),
		       np.random.uniform(sigma_hub[0], sigma_hub[1]),
		       np.random.uniform(sigma_tip[0], sigma_tip[1]),
		       np.random.uniform(nu[0], nu[1]),
		       np.random.uniform(tip_clearance[0], tip_clearance[1]),
		       np.random.choice(airfoil_dist)]
		list_turb_params.append(turb_params)

	return list_turb_params

# FUNCIÓN 2
# Parámetros:
#   >N: parámetro originariamente constante, y del cual se quiere modificar su precisión.
# Devuelve: un diccionario con todos los parámetros constantes que será necesario para hacer una 
# evaluación.

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

	# Construir el diccionario que necesita la función fitness
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

# FUNCIÓN 3
# Parámetros:
#   >turb_params: lista que representa una configuración de los parámetros que definen el
#	 diseño de las turbinas.
#	>N: parámetro originariamente constante, y del cual se quiere modificar su precisión.
# Devuelve: scores por individual (de cada estado de mar) y escore total (la suma ponderada de
# los anteriores), asociado a la evaluación del diseño turb_params. 

def evaluation(turb_params,N=50):

	# Construir diccionario de parámetros constantes.
	constargs=build_constargs_dict(N)

	# Crear turbina instantantanea.
	os.chdir('OptimizationAlgorithms_KONFLOT')
	turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
	os.chdir('../')

	# Calcular evaluación.
	scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

	total_fitness=scores[1]
	partial_fitness=[]
	for e, score in enumerate(scores[0]):
		partial_fitness.append(score)

	return total_fitness,partial_fitness

# FUNCIÓN 4
# Parámetros:
#   >set_turb_params: lista de listas que representan configuraciones de los parámetros 
#    que definen el diseño de las turbinas.
#	>N: parámetro originariamente constante, y del cual se quiere modificar su precisión.
# Devuelve: una lista con información asociada a la evaluación de todas las configuraciones/diseños
# que forman set_turb_params para el N fijado.

def set_evaluation(set_turb_params,N,accuracy=None):

	# Calcular scores y tiempos de ejecución para la evaluación de cada configuración.
	all_scores=[]
	all_times=[]
	n_solution=0
	for turb_param in set_turb_params:

		t=time.time()
		total_fitness,partial_fitness=evaluation(turb_param,n)
		elapsed=time.time()-t
		n_solution+=1

		all_scores.append(total_fitness)
		all_times.append(elapsed)
		df.append([accuracy,n_solution,total_fitness,elapsed])

	# Calcular datos que se guardarán en la base de datos.
	ranking=from_argsort_to_ranking(np.argsort(all_scores))
	total_time=sum(all_times)
	time_per_eval=np.mean(all_times)


	return [N,all_scores,ranking,all_times,total_time,time_per_eval]

	
# FUNCIÓN 5
# Parámetros:
#   >list: lista conseguida tras aplicar "np.argsort" sobre una lista (original).
# Devolver: nueva lista que representa el ranking de los elementos de la lista original.
def from_argsort_to_ranking(list):
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

#--------------------------------------------------------------------------------------------------
# Para el análisis de motivación (PRIMER ANÁLISIS).
#--------------------------------------------------------------------------------------------------
# Escoger aleatoriamente una conjunto de configuraciones.
list_seeds=range(0,10)#Fijar semillas para la reproducibilidad
set_turb_params = choose_random_configurations(list_seeds)

# Mallado para N.
grid_N=[1000,900,800,700,600,500,400,300,200,100,50,45,40,35,30,25,20,15,10,5]

# Guardar en una base de datos los datos asociados a la evaluación del conjunto de configuraciones/
# diseños para cada valor de N considerado.
df=[]
for n in tqdm(grid_N):
	df.append(set_evaluation(set_turb_params,n))

df=pd.DataFrame(df,columns=['N','all_scores','ranking','all_times','total_time','time_per_eval'])
df.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyI.csv')

#--------------------------------------------------------------------------------------------------
# Para el análisis de motivación (SEGUNDO ANÁLISIS).
#--------------------------------------------------------------------------------------------------
# Escoger aleatoriamente una conjunto de configuraciones.
list_seeds=range(0,100)#Fijar semillas para la reproducibilidad
set_turb_params = choose_random_configurations(list_seeds)

# Mallado para N.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]

# Guardar en una base de datos los datos asociados a la evaluación del conjunto de configuraciones/
# diseños para cada valor de N considerado.
default_N=50
df=[]
for acc in tqdm(list_acc):
	n=int(default_N*acc)
	set_evaluation(set_turb_params,n,accuracy=acc)

df=pd.DataFrame(df,columns=['accuracy','n_solution','score','time'])
df.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII.csv')

#--------------------------------------------------------------------------------------------------
# Para la fijación del tamaño de muestra del método de bisección.
#--------------------------------------------------------------------------------------------------
# Lista con los valores de accuracy que se considerarían por el método de bisección, teniendo en
# cuenta que el criterio de parada es alcanzar un rango del intervalo de 0.1 y suponiendo que
# en todas las iteraciones se acota el intervalo por arriba (caso más costoso).
def upper_middle_point(lower,upper=1.0):
    list=[] 
    while abs(lower-upper)>0.1:       
        middle=(lower+upper)/2
        list.append(middle)
        lower=middle
    return list
list_acc=upper_middle_point(10/default_N)+[1.0]

# Evaluar una muestra aleatoria usando los valores anteriores de accuracy.
list_seeds=range(0,100)#Fijar semillas para la reproducibilidad
set_turb_params = choose_random_configurations(list_seeds)
df=[]
for acc in tqdm(list_acc):
	n=int(default_N*acc)
	set_evaluation(set_turb_params,n,accuracy=acc)

df=pd.DataFrame(df,columns=['accuracy','n_solution','score','cost_per_eval'])
df=df[['accuracy','cost_per_eval']]
df=df.groupby('accuracy').mean()
df.to_csv('results/data/Turbines/UnderstandingAccuracy/df_BisectionSample.csv')


