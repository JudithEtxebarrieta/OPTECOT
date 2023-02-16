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
import scipy as sc
import random
from itertools import combinations
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
#--------------------------------------------------------------------------------------------------
# Funciones auxiliares para definir el accuracy apropiado en cada momento del proceso.
#--------------------------------------------------------------------------------------------------
# Cálculo de la correlación de Spearman entre dos secuencias.
def spearman_corr(x,y):
    return sc.stats.spearmanr(x,y)[0]

# Cálculo dela distancia Tau Kendall inversa normalizada entre dos secuencias.
def inverse_normalized_tau_kendall(x,y):
    # Número de pares con orden inverso.
    pairs_reverse_order=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            case1 = x[i] < x[j] and y[i] > y[j]
            case2 = x[i] > x[j] and y[i] < y[j]

            if case1 or case2:
                pairs_reverse_order+=1  
    
    # Número de pares total.
    total_pairs=len(list(combinations(x,2)))

    # Distancia tau Kendall normalizada.
    tau_kendall=pairs_reverse_order/total_pairs

    return 1-tau_kendall

# Convertir vector de scores en vector de rankings.
def from_scores_to_ranking(list_scores):
    list_pos_ranking=np.argsort(np.array(list_scores))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

# Generar lista con los scores asociados a cada superficie que forma la generación.
def generation_score_list(population,accuracy,count_time_acc=True,count_time_gen=False):

    # Generar entorno con accuracy apropiado.
    windFLO=get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)

    # Evaluar población.
    list_scores=[]
    

    for sol in population:
        if count_time_acc and not count_time_gen:
            sw_acc.resume()
        if count_time_gen:
            sw.resume()
        score=EvaluateFarm(sol,windFLO)
        if count_time_acc and not count_time_gen:
            sw_acc.pause()
        if count_time_gen:
            sw.pause()
        list_scores.append(score)

    if count_time_gen:
        return list_scores,windFLO
    else:
        return list_scores

#--------------------------------------------------------------------------------------------------
# Funciones asociadas al heurístico que se aplicará para ajustar el accuracy..
#--------------------------------------------------------------------------------------------------
# Implementación adaptada del método de bisección.
def bisection_method(init_acc,population,train_seed,metric,threshold=0.95):

    # Inicializar límite inferior y superior.
    acc0=init_acc
    acc1=1    

    # Punto intermedio.
    prev_m=init_acc
    m=(acc0+acc1)/2
    
    # Función para calcular la correlación entre los rankings del 10% aleatorio/mejor de las superficies
    # usando el accuracy actual y el máximo.
    def similarity_between_current_best_acc(acc,population,train_seed,metric,first_iteration):

        # Seleccionar de forma aleatoria el 10% de las superficies que forman la generación.
        random.seed(train_seed)
        ind_sol=random.sample(range(len(population)),int(len(population)*0.1))
        list_solutions=list(np.array(population)[ind_sol])

        # Guardar los scores asociados a cada solución seleccionada.
        previous_time_acc=sw_acc.get_time()
        best_scores=generation_score_list(list_solutions,1,count_time_acc=first_iteration)# Con el máximo accuracy. 
        new_scores=generation_score_list(list_solutions,acc)# Accuracy nuevo. 
        last_time_acc_increase=sw_acc.get_time()-previous_time_acc

        # Obtener vectores de rankings asociados.
        new_ranking=from_scores_to_ranking(new_scores)# Accuracy nuevo. 
        best_ranking=from_scores_to_ranking(best_scores)# Máximo accuracy. 
                
        # Comparar ambos rankings.
        if metric=='spearman':
            metric_value=spearman_corr(new_ranking,best_ranking)
        if metric=='taukendall':
            metric_value=inverse_normalized_tau_kendall(new_ranking,best_ranking)

        return metric_value,last_time_acc_increase

    # Reajustar límites del intervalo hasta que este tenga un rango lo suficientemente pequeño.
    first_iteration=True

    while acc1-acc0>0.1:
        metric_value,last_time_acc_increase=similarity_between_current_best_acc(m,population,train_seed,metric,first_iteration)
        if metric_value>=threshold:
            acc1=m
        else:
            acc0=m

        prev_m=m
        m=(acc0+acc1)/2
        
        first_iteration=False

    return prev_m,last_time_acc_increase

def execute_heuristic(gen,min_acc,previous_acc,population,train_seed,param):
    global last_optimal_time
    global sw
    global sw_acc

    if gen==0:
        acc,time_best_acc=bisection_method(min_acc,population,train_seed,param)
        last_optimal_time=sw.get_time()+sw_acc.get_time()

    else:
        if (sw.get_time()+sw_acc.get_time())-last_optimal_time>=max_time*0.1:
            acc,time_best_acc=bisection_method(min_acc,population,train_seed,param)
            last_optimal_time=sw.get_time()+sw_acc.get_time()
        else:
            acc=previous_acc
            time_best_acc=0


    return acc,time_best_acc

#--------------------------------------------------------------------------------------------------
# Funciones para el proceso de aprendizaje o búsqueda de la óptima distribución de molinos.
#--------------------------------------------------------------------------------------------------

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
def learn(seed,heuristic_param,maxfeval=500,popsize=50): 

    global folder_name
    folder_name='File_'+str(heuristic_param)
    os.makedirs(folder_name)


    # Inicializar el terreno y las turbinas que se desean colocar sobre el mismo.
    windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/')

    # Accuracy mínimo.
    min_acc=1/windFLO.montecarlopts
    
    # Función para transformar el valor escalado de los parámetros en los valores reales.
    def transform_to_problem_dim(list_coord):
        lbound = np.zeros(windFLO.nTurbines*2) # Límite inferior real.
        ubound = np.ones(windFLO.nTurbines*2)*2000 # Límite superior real.
        return lbound + list_coord*(ubound - lbound)

    # Inicializar contador de tiempo.
    global sw
    sw = stopwatch()
    sw.pause()

    global sw_acc
    sw_acc = stopwatch()
    sw_acc.pause()

    sw_acc_duplicate=0
    n_gen=0
    accuracy=None# Para la primera generación.

    # Aplicar algoritmo CMA-ES para la búsqueda de la solución.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(windFLO.nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfeval, 'popsize':popsize})
    
    while sw.get_time()<max_time:

        # Construir generación.
        solutions = es.ask()

        # Transformar los valores escalados de los parámetros a los valores reales.
        real_solutions=[transform_to_problem_dim(list_coord) for list_coord in solutions]

        # Reajustar el accuracy según el heurístico seleccionado.
        accuracy,time_best_acc_bisection=execute_heuristic(n_gen,min_acc,accuracy,real_solutions,seed,heuristic_param)
        sw_acc_duplicate+=time_best_acc_bisection

        # Lista de scores asociados a la generación.
        list_scores,windFLO=generation_score_list(real_solutions,accuracy,count_time_gen=True)

        # Para construir la siguiente generación.
        es.tell(solutions, list_scores)

        # Acumular datos de interés.
        score = EvaluateFarm(transform_to_problem_dim(es.result.xbest),windFLO)
        df.append([accuracy,seed,n_gen,-score,sw.get_time(),sw_acc.get_time()-sw_acc_duplicate,sw.get_time()+sw_acc.get_time()-sw_acc_duplicate])

        n_gen+=1

    os.rmdir(folder_name)



#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Lista de semillas de entrenamiento.
list_seeds=range(1,51,1)

# Lista de parámetros del heurístico.
list_param=['spearman','taukendall']

# Máximo tiempo de ejecución.
max_time=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/max_time.npy')

# Función para realizar la ejecución en paralelo.
def parallel_processing(arg):
    global df
    df=[]

    for seed in tqdm(list_seeds):
        learn(seed,arg)
    
    df=pd.DataFrame(df,columns=['accuracy','seed','n_gen','score','elapsed_time_proc','elapsed_time_acc','total_elapsed_time'])
    df.to_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_'+str(arg)+'.csv')



# Ejecución en paralelo para el resto de accuracys.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,list_param)
pool.close()


# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')