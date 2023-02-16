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

# Función para inicializar las características del terreno y las turbinas sobre los cuales se aplicará la optimización.
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



# Función para evaluar el desempeño del diseño del parque eólico.
def EvaluateFarm(x, windFLO):
    
    k = 0
    for i in range(0, windFLO.nTurbines):
        for j in range(0, 2):
            # unroll the variable vector 'x' and assign it to turbine positions
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1

    # Eliminar ficheros auxiliares.
    windFLO.run(clean = True) 
    windFLO.run(clean = True, inFile = 'WindFLO.res')
    windFLO.run(clean = True, inFile = 'terrain.dat')

    return -windFLO.farmPower


# Función para aprender la solución.
def learn(seed, accuracy,maxfeval=500,popsize=50): 

    global max_n_eval
    max_n_eval=maxfeval 


    # Inicializar el terreno y las turbinas que se desean colocar sobre el mismo.
    folder_name='File'+str(accuracy)
    os.makedirs(folder_name)
    windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)
    
    # Función para transformar el valor escalado de los parámetros en los valores reales.
    def transform_to_problem_dim(list_coord):
        lbound = np.zeros(windFLO.nTurbines*2) # Límite inferior real.
        ubound = np.ones(windFLO.nTurbines*2)*2000 # Límite superior real.
        return lbound + list_coord*(ubound - lbound)

    # Inicializar contador de tiempo.
    global sw
    sw = stopwatch()
    sw.pause()

    global n_evaluations
    n_evaluations=0

    n_gen=0

    # Aplicar algoritmo CMA-ES para la búsqueda de la solución.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(windFLO.nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfeval, 'popsize':popsize})
    
    while not es.stop():

        # Construir generación.
        solutions = es.ask()

        # Transformar los valores escalados de los parámetros a los valores reales.
        real_solutions=[transform_to_problem_dim(list_coord) for list_coord in solutions]

        # Lista de scores asociados a la generación.
        list_scores=[]
        for sol in real_solutions:

            sw.resume()
            fitness=EvaluateFarm(sol,windFLO)
            sw.pause()

            list_scores.append(fitness)
            n_evaluations+=1

        # Para construir la siguiente generación.
        es.tell(solutions, list_scores)

        # Acumular datos de interés.
        score = EvaluateFarm(transform_to_problem_dim(es.result.xbest),windFLO)
        df_acc.append([accuracy,seed,n_gen,-score,sw.get_time()])

        n_gen+=1
  
    os.rmdir(folder_name)

    if accuracy==1:
        return sw.get_time()


def new_stop_max_acc(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
    stop={}
    if n_evaluations>max_n_eval:
        stop={'TIME RUN OUT':max_n_eval}
    return stop

def new_stop_lower_acc(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
	stop={}
	if sw.get_time()>max_time:
		stop={'TIME RUN OUT':max_time}
	return stop

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Lista de semillas de entrenamiento.
list_seeds=range(1,51,1)

# Lista de accuracys a considerar.
list_acc=[1.0,0.5,0.2,0.1,0.05,0.02,0.01,0.005]

# El caso de accuracy 1 hay que ejecutarlo el primero para definir el límite de tiempo de 
# ejecución para el resto.
def process_max_acc():

    global df_acc
    df_acc=[]

    cma.CMAEvolutionStrategy.stop=new_stop_max_acc
    list_total_time=[]

    for seed in tqdm(list_seeds):
        total_time=learn(seed,list_acc[0])
        list_total_time.append(total_time)

    df_acc=pd.DataFrame(df_acc,columns=['accuracy','seed','n_gen','score','elapsed_time'])
    df_acc.to_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis'+str(list_acc[0])+'.csv')

    return min(list_total_time)

# Función para realizar la ejecución en paralelo.
def parallel_processing(arg):

    global df_acc
    df_acc=[]

    cma.CMAEvolutionStrategy.stop=new_stop_lower_acc

    for seed in tqdm(list_seeds):
        learn(seed,arg)
    
    df_acc=pd.DataFrame(df_acc,columns=['accuracy','seed','n_gen','score','elapsed_time'])
    df_acc.to_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis'+str(arg)+'.csv')

# Ejecución para accuracy 1.
max_time=process_max_acc()
# df=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv", index_col=0)
# max_time=min(df.groupby('seed')['elapsed_time'].max())
np.save('results/data/WindFLO/ConstantAccuracyAnalysis/max_time',max_time)


# Ejecución en paralelo para el resto de accuracys.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,list_acc[1:])
pool.close()

# Guardar lista con valores de accuracy.
np.save('results/data/WindFLO/ConstantAccuracyAnalysis/list_acc',list_acc)

# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')