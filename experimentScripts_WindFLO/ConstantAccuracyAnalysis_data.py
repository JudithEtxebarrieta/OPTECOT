# Mediante este script se aplica el algoritmo CMA-ES durante un tiempo máximo de ejecución, 
# considerando 10 valores diferentes de accuracy y 50 semillas para cada uno de ellos. Primero se 
# ejecuta el algoritmo para accuracy 1, fijando así el límite de tiempo de entrenamiento en el 
# mínimo tiempo entre los tiempos totales necesarios para cada semilla. Por cada valor de accuracy
# se construirá una base de datos con la información relevante durante en entrenamiento.

#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import cma
from tqdm import tqdm as tqdm
import pandas as pd
import multiprocessing as mp


sys.path.append('WindFLO/API')
from WindFLO import WindFLO


#==================================================================================================
# CLASES
#==================================================================================================
class stopwatch:
    
    def __init__(self):
        self.reset()
        self.paused = False

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

# FUNCIÓN 1 (Inicializar las características del terreno y las turbinas sobre los cuales se aplicará la optimización)
def get_windFLO_with_accuracy(momentary_folder='',accuracy=1):

    # Configuración y parámetros de WindFLO.
    windFLO = WindFLO(
    inputFile = 'WindFLO/Examples/Example1/WindFLO.dat', # Archivo input para leer.
    libDir = 'WindFLO/release/', # Ruta a la librería compartida libWindFLO.so.
    turbineFile = 'WindFLO/Examples/Example1/V90-3MW.dat',# Parámetros de las turbinas.
    terrainfile = 'WindFLO/Examples/Example1/terrain.dat', # Archivo del terreno.
    runDir=momentary_folder,
    nTurbines = 25, # Número de turbinas.

    monteCarloPts = round(1000*accuracy)# Parámetro del cual se modificará su precisión.
    )

    # Cambiar el modelo de terreno predeterminado de RBF a IDW.
    windFLO.terrainmodel = 'IDW'

    return windFLO

# FUNCIÓN 2 (Evaluar el desempeño del diseño del parque eólico)
def EvaluateFarm(x, windFLO):
    
    k = 0
    for i in range(0, windFLO.nTurbines):
        for j in range(0, 2):
            # unroll the variable vector 'x' and assign it to turbine positions
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1

    # Run WindFLO analysis
    windFLO.run(clean = True) 

    return -windFLO.farmPower

# FUNCIÓN 3 (Buscar la solución óptima aplicando el algoritmo CMA-ES)
def learn(seed, accuracy,maxfeval=500,popsize=50): 

    global max_n_eval
    max_n_eval=maxfeval 


    # Inicializar el terreno y las turbinas que se desean colocar sobre el mismo.
    folder_name='File'+str(accuracy)
    os.makedirs(folder_name)
    windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)
    default_windFLO= get_windFLO_with_accuracy(momentary_folder=folder_name+'/')
    
    # Función para transformar el valor escalado de los parámetros en los valores reales.
    def transform_to_problem_dim(list_coord):
        lbound = np.zeros(windFLO.nTurbines*2) # Límite inferior real.
        ubound = np.ones(windFLO.nTurbines*2)*2000 # Límite superior real.
        return lbound + list_coord*(ubound - lbound)

    # Inicializar contador de tiempo.
    global eval_time
    eval_time=0

    global n_evaluations
    n_evaluations=0

    n_gen=0

    # Aplicar algoritmo CMA-ES para la búsqueda de la solución.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(default_windFLO.nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfeval, 'popsize':popsize})
    
    while not es.stop():

        # Construir generación.
        solutions = es.ask()

        # Transformar los valores escalados de los parámetros a los valores reales.
        real_solutions=[transform_to_problem_dim(list_coord) for list_coord in solutions]

        # Lista de scores asociados a la generación.
        list_scores=[]
        for sol in real_solutions:

            t=time.time()
            fitness=EvaluateFarm(sol,windFLO)
            eval_time+=time.time()-t

            list_scores.append(fitness)
            n_evaluations+=1

        # Para construir la siguiente generación.
        es.tell(solutions, list_scores)

        # Acumular datos de interés.
        score = EvaluateFarm(transform_to_problem_dim(es.result.xbest),default_windFLO)
        df_acc.append([accuracy,seed,n_gen,-score,eval_time])

        n_gen+=1
  
    os.rmdir(folder_name)

    if accuracy==1:
        return eval_time

# FUNCIÓN 4 (Criterio de parada para accuracy=1)
def new_stop_max_acc(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
    stop={}
    if n_evaluations>max_n_eval:
        stop={'TIME RUN OUT':max_n_eval}
    return stop

# FUNCIÓN 5 (Criterio de parada para accuracys menores)
def new_stop_lower_acc(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
	stop={}
	if eval_time>max_time:
		stop={'TIME RUN OUT':max_time}
	return stop

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Lista de semillas de entrenamiento.
list_seeds=range(1,51,1)

# Lista de accuracys a considerar.
list_acc=[1.0,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]

# Construir base de datos con datos relevantes por cada ejecución con un valor de accuracy.
for accuracy in list_acc:

    global df_acc
    df_acc=[]

    # El caso de accuracy 1 hay que ejecutarlo el primero para definir el límite de tiempo de 
    # ejecución para el resto.
    if accuracy==1:
        cma.CMAEvolutionStrategy.stop=new_stop_max_acc
        list_total_time=[]

    # Para el resto de accuracys.
    else:
        cma.CMAEvolutionStrategy.stop=new_stop_lower_acc

    for seed in tqdm(list_seeds):
        total_time=learn(seed,accuracy)
        list_total_time.append(total_time)

    df_acc=pd.DataFrame(df_acc,columns=['accuracy','seed','n_gen','score','elapsed_time'])
    df_acc.to_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis'+str(accuracy)+'.csv')

    if accuracy==1:
        max_time=min(list_total_time)
        np.save('results/data/WindFLO/ConstantAccuracyAnalysis/max_time',max_time)

# Guardar lista con valores de accuracy.
np.save('results/data/WindFLO/ConstantAccuracyAnalysis/list_acc',list_acc)

# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')
