# Mediante este script se guarda en una base de datos la informacion relevante asociada a la 
# evaluacion de un conjunto de configuraciones/disenos de turbinas. El conjunto esta formado por
# un total de 10 o 100 disenos de turbina escogidos de forma aleatoria, y todas las configuraciones del
# mismo se evaluan considerando diferentes valores de N. Se construiran unas bases de datos con la 
# informacion de scores y tiempos de ejecucion por evaluacion.

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

import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")
import turbine_classes
import MathTools as mt

from tqdm import tqdm
import pandas as pd

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCION 1
# Parametros:
#   >list_seeds: lista de semillas a partir de las cuales se seleccionaran el
#    conjunto de turbinas aleatorias, (tantas turbinas como cardinal de esta lista).
# Devuelve: lista con configuraciones aleatorias de los parametros que definen el diseno 
# de las turbinas.

def choose_random_configurations(list_seeds,blade_number=[3,5,7]):
	# Definir rangos de los parametros que definen el diseno de la turbina.
	blade_number = blade_number# Blade-number gene.
	sigma_hub = [0.4, 0.7]# Hub solidity gene.
	sigma_tip = [0.4, 0.7]# Tip solidity gene.
	nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
	tip_clearance=[0,3]# Tip-clearance gene.	  
	airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

	# Guardar tantas configuraciones aleatorias como semillas en list_seeds.
	list_turb_params=[]
	for seed in list_seeds: 
		np.random.seed(seed)
		if type(blade_number)==list:
			select_blade_number=np.random.choice(blade_number)
		if type(blade_number)==int:
			select_blade_number=blade_number
		turb_params = [select_blade_number,
		       np.random.uniform(sigma_hub[0], sigma_hub[1]),
		       np.random.uniform(sigma_tip[0], sigma_tip[1]),
		       np.random.uniform(nu[0], nu[1]),
		       np.random.uniform(tip_clearance[0], tip_clearance[1]),
		       np.random.choice(airfoil_dist)]
		list_turb_params.append(turb_params)

	return list_turb_params

# FUNCION 2
# Parametros:
#   >N: parametro originariamente constante, y del cual se quiere modificar su precision.
# Devuelve: un diccionario con todos los parametros constantes que sera necesario para hacer una 
# evaluacion.

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

# FUNCION 3
# Parametros:
#   >turb_params: lista que representa una configuracion de los parametros que definen el
#	 diseno de las turbinas.
#	>N: parametro originariamente constante, y del cual se quiere modificar su precision.
# Devuelve: scores por individual (de cada estado de mar) y escore total (la suma ponderada de
# los anteriores), asociado a la evaluacion del diseno turb_params. 

def evaluation(turb_params,N=100):

	# Construir diccionario de parametros constantes.
	constargs=build_constargs_dict(N)

	# Crear turbina instantantanea.
	os.chdir('OptimizationAlgorithms_KONFLOT')
	turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
	os.chdir('../')

	# Calcular evaluacion.
	scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

	total_fitness=scores[1]
	partial_fitness=[]
	for e, score in enumerate(scores[0]):
		partial_fitness.append(score)

	return total_fitness,partial_fitness

# FUNCION 4
# Parametros:
#   >set_turb_params: lista de listas que representan configuraciones de los parametros 
#    que definen el diseno de las turbinas.
#	>N: parametro originariamente constante, y del cual se quiere modificar su precision.
# Devuelve: una lista con informacion asociada a la evaluacion de todas las configuraciones/disenos
# que forman set_turb_params para el N fijado.

def set_evaluation(set_turb_params,N,accuracy=None):

	# Calcular scores y tiempos de ejecucion para la evaluacion de cada configuracion.
	all_scores=[]
	all_times=[]
	n_solution=0
	for turb_param in tqdm(set_turb_params):

		t=time.time()
		total_fitness,partial_fitness=evaluation(turb_param,N=N)
		elapsed=time.time()-t
		n_solution+=1

		all_scores.append(total_fitness)
		all_times.append(elapsed)
		if accuracy!=None:
			df.append([accuracy,n_solution,total_fitness,elapsed])

	# Calcular datos que se guardaran en la base de datos.
	ranking=from_argsort_to_ranking(np.argsort(all_scores))
	total_time=sum(all_times)
	time_per_eval=np.mean(all_times)

	if accuracy==None:
		return [N,all_scores,ranking,all_times,total_time,time_per_eval]

	
# FUNCION 5
# Parametros:
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
# Para el analisis de motivacion (PRIMER ANALISIS).
#--------------------------------------------------------------------------------------------------
# Escoger aleatoriamente una conjunto de configuraciones.
list_seeds=range(0,10)#Fijar semillas para la reproducibilidad
set_turb_params = choose_random_configurations(list_seeds)

# Mallado para N.
grid_N=[1000,900,800,700,600,500,400,300,200,100,50,45,40,35,30,25,20,15,10,5]

# Guardar en una base de datos los datos asociados a la evaluacion del conjunto de configuraciones/
# disenos para cada valor de N considerado.
df=[]
for n in tqdm(grid_N):
	df.append(set_evaluation(set_turb_params,n))

df=pd.DataFrame(df,columns=['N','all_scores','ranking','all_times','total_time','time_per_eval'])
df.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyI.csv')

#--------------------------------------------------------------------------------------------------
# Para el analisis de motivacion (SEGUNDO ANALISIS).
#--------------------------------------------------------------------------------------------------

# Analisis para seleccionar la definicion de blade-number que se considerara.
for blade_number in [3,5,7,[3,5,7]]:
	# Escoger aleatoriamente una conjunto de configuraciones.
	list_seeds=range(0,100)#Fijar semillas para la reproducibilidad
	set_turb_params = choose_random_configurations(list_seeds,blade_number=blade_number)

	# Mallado para N.
	list_n=range(5,101,1)
	list_acc=[]
	for n in list_n:
		list_acc.append(n/100)
	list_acc=[1]
	# Guardar en una base de datos los datos asociados a la evaluacion del conjunto de soluciones
	# para cada valor de N considerado.
	default_N=100
	df=[]
	for acc in tqdm(list_acc):
		set_evaluation(set_turb_params,int(default_N*acc),accuracy=acc)

	df_motivation=pd.DataFrame(df,columns=['accuracy','n_solution','score','time'])
	if type(blade_number)==list:
		df_motivation.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumberAll.csv')
	else:
		df_motivation.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber'+str(blade_number)+'.csv')

# Reducir base de datos del blade-number fijado a los valores de accuracy de interes.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
df=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumberAll.csv',index_col=0)
df=df.iloc[[i in list_acc for i in df['accuracy']]]
df.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII.csv')

#--------------------------------------------------------------------------------------------------
# Para la definicion de los valores (tiempo) sobre los cuales se aplicara la biseccion.
#--------------------------------------------------------------------------------------------------
# Guardar base de datos.
df.columns=['accuracy','n_solution','score','cost_per_eval']
df_bisection=df[['accuracy','cost_per_eval']]
df_bisection=df_bisection.groupby('accuracy').mean()
df_bisection.to_csv('results/data/Turbines/UnderstandingAccuracy/df_Bisection.csv')




