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
# FUNCIONES
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Funciones auxiliares para definir el accuracy apropiado en cada momento del proceso.
#--------------------------------------------------------------------------------------------------

# FUNCIÓN 1 (Cálculo de la correlación de Spearman entre dos secuencias)
def spearman_corr(x,y):
    return sc.stats.spearmanr(x,y)[0]

# FUNCIÓN 2 (Convertir vector de scores en vector de ranking)
def from_scores_to_ranking(list_scores):
    list_pos_ranking=np.argsort(np.array(list_scores))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

# FUNCIÓN 3 (Generar lista con los scores asociados a cada superficie que forma la generación)
# Parámetros:
#   >population: lista con las soluciones que forman la generación.
#   >accuracy: accuracy fijado como óptimo en la generación anterior.
#   >count_time_acc: True o False si se quieren sumar o no respectivamente, el tiempo de evaluación
#    como tiempo adicional para ajustar el accuracy.
#   >count_time_gen: True o False si se quiere sumar o no respectivamente, el tiempo de evaluación
#    como tiempo natural para la evaluación de la generación.
# Devolver: lista con los scores asociados a cada solución que forma la generación.

def generation_score_list(population,accuracy,count_time_acc=True,count_time_gen=False):
    global time_acc,time_proc

    # Generar entorno con accuracy apropiado.
    windFLO=get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)

    # Evaluar población.
    list_scores=[]
    for sol in population:
        t=time.time()
        score=EvaluateFarm(sol,windFLO)
        elapsed_time=time.time()-t
        if count_time_acc and not count_time_gen:
            time_acc+=elapsed_time
        if count_time_gen:
            time_proc+=elapsed_time
        list_scores.append(score)

    return list_scores

#--------------------------------------------------------------------------------------------------
# Funciones asociadas a los heurísticos que se aplicarán para ajustar el accuracy..
#--------------------------------------------------------------------------------------------------
# FUNCIÓN 4 (Implementación adaptada del método de bisección)
# Parámetros:
#   >init_acc: accuracy inicial (el mínimo considerado).
#   >population: lista con las soluciones que forman la generación.
#   >train_seed: semilla de entrenamiento.
#   >threshold: umbral del método de bisección con el que se irá actualizando el intervalo que contiene el valor de accuracy óptimo.
# Devuelve: 
#   >prev_m: valor de accuracy seleccionado como óptimo.
#   >last_time_acc_increase: tiempo de evaluación consumido en la última iteración del método de bisección.

def bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_expression,threshold=0.95):

    # Inicializar límite inferior y superior.
    time0=lower_time
    time1=upper_time   

    # Punto intermedio.
    prev_m=time0
    m=(time0+time1)/2

    # Función para calcular la correlación entre los rankings de las sample_size superficies aleatorias
    # usando el accuracy actual y el máximo.
    def similarity_between_current_best_acc(acc,population,train_seed,first_iteration):
        global time_acc

        # Seleccionar de forma aleatoria sample_size superficies que forman la generación.
        random.seed(train_seed)
        ind_sol=random.sample(range(len(population)),sample_size)
        list_solutions=list(np.array(population)[ind_sol])

        # Guardar los scores asociados a cada solución seleccionada.
        t=time.time()
        best_scores=generation_score_list(list_solutions,1,count_time_acc=first_iteration)# Con el máximo accuracy. 
        new_scores=generation_score_list(list_solutions,acc)# Accuracy nuevo. 
        last_time_acc_increase=time.time()-t

        # Obtener vectores de rankings asociados.
        new_ranking=from_scores_to_ranking(new_scores)# Accuracy nuevo. 
        best_ranking=from_scores_to_ranking(best_scores)# Máximo accuracy. 
                
        # Comparar ambos rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)

        return metric_value,last_time_acc_increase

    # Reajustar límites del intervalo hasta que este tenga un rango lo suficientemente pequeño.
    first_iteration=True
    stop_threshold=(time1-time0)*0.1
    while time1-time0>stop_threshold:
        metric_value,last_time_acc_increase=similarity_between_current_best_acc(eval(interpolation_expression),population,train_seed,first_iteration)
        if metric_value>=threshold:
            time1=m
        else:
            time0=m

        prev_m=m
        m=(time0+time1)/2

        first_iteration=False
    m=prev_m
    return eval(interpolation_expression),last_time_acc_increase

# FUNCIÓN 5 (Ejecutar heurísticos durante el proceso de entrenamiento)
# Parámetros:
#   >gen: número de generación en el algoritmo CMA-ES.
#   >min_acc: accuracy mínimo a considerar en el método de bisección.
#   >acc: accuracy asociado a la generación anterior.
#   >population: lista con las soluciones que forman la generación.
#   >train_seed: semilla de entrenamiento.
#   >list_variances: lista con las varianzas de los scores de las anteriores generaciones.
#   >heuristic: número que identifica al heurístico que se desea considerar.
#   >param: valor del parámetro asociado al heurístico que se desea aplicar.
# Devuelve: 
#   >acc: valor de accuracy seleccionado como óptimo.
#   >time_best_acc: tiempo de evaluación consumido en la última iteración del método de bisección.

def execute_heuristic(gen,acc,population,train_seed,list_variances,heuristic,param):

    global time_proc
    global time_acc
    global max_time
    global last_time_heuristic_accepted
    global unused_bisection_executions

    heuristic_accepted=False

    # Para el método de bisección: tamaño de muestra, frecuencia y expresión de interpolación.
    df_sample_freq=pd.read_csv('results/data/general/sample_size_freq_'+str(sample_size_freq)+'.csv',index_col=0)
    df_interpolation=pd.read_csv('results/data/general/bisection_interval_interpolation.csv',index_col=0)
    sample_size=int(df_sample_freq[df_sample_freq['env_name']=='WindFLO']['sample_size'])
    heuristic_freq=float(df_sample_freq[df_sample_freq['env_name']=='WindFLO']['frequency_time'])
    interpolation_expression=list(df_interpolation[df_interpolation['env_name']=='WindFLO']['interpolation_expression'])[-1]
    lower_time=float(df_interpolation[df_interpolation['env_name']=='WindFLO']['lower_time'])
    upper_time=float(df_interpolation[df_interpolation['env_name']=='WindFLO']['upper_time'])

   
    # HEURÍSTICO I de Symbolic Regressor: Bisección de generación en generación (el umbral es el parámetro).
    if heuristic=='I': 
        if gen==0:
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_expression,threshold=param)
            list_scores=generation_score_list(population,acc,count_time_gen=True)
            time_acc-=time_best_acc
            last_time_heuristic_accepted=time_proc+time_acc
            heuristic_accepted=True
            
        else:
            if (time_acc+time_proc)-last_time_heuristic_accepted>=heuristic_freq:
                acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_expression,threshold=param)
                list_scores=generation_score_list(population,acc,count_time_gen=True)
                time_acc-=time_best_acc
                last_time_heuristic_accepted=time_proc+time_acc
                heuristic_accepted=True


    # HEURÍSTICO II de Symbolic Regressor: Bisección con definición automática para frecuencia 
    # de actualización de accuracy (depende de parámetro) y umbral del método de bisección fijado en 0.95.
    if heuristic=='II': 
        if gen==0: 
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_expression)
            list_scores=generation_score_list(population,acc,count_time_gen=True)
            time_acc-=time_best_acc
            last_time_heuristic_accepted=time_proc+time_acc
            unused_bisection_executions=0
            heuristic_accepted=True
        else:
            if len(list_variances)>=param+1:
                # Función para calcular el intervalo de confianza.
                def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                    mean_list=[]
                    for i in range(bootstrap_iterations):
                        sample = np.random.choice(data, len(data), replace=True) 
                        mean_list.append(np.mean(sample))
                    return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

                variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-param):-2])
                last_variance=list_variances[-1]
                
                # Calcular el mínimo accuracy con el que se obtiene la máxima calidad.
                if last_variance<variance_q05 or last_variance>variance_q95:

                    if (time_proc+time_acc)-last_time_heuristic_accepted>=heuristic_freq:   
                        unused_bisection_executions+=int((time_proc+time_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                        acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_expression)
                        list_scores=generation_score_list(population,acc,count_time_gen=True)
                        time_acc-=time_best_acc
                        last_time_heuristic_accepted=time_proc+time_acc
                        heuristic_accepted=True

                    else:
                        if unused_bisection_executions>0:
                            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_expression)
                            list_scores=generation_score_list(population,acc,count_time_gen=True)
                            time_acc-=time_best_acc
                            last_time_heuristic_accepted=time_proc+time_acc
                            unused_bisection_executions-=1
                            heuristic_accepted=True
            else:
                if (time_acc+time_proc)-last_time_heuristic_accepted>=heuristic_freq:
                    acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_expression,threshold=param)
                    list_scores=generation_score_list(population,acc,count_time_gen=True)
                    time_acc-=time_best_acc
                    last_time_heuristic_accepted=time_proc+time_acc
                    heuristic_accepted=True

    if heuristic_accepted==False:
        list_scores=generation_score_list(population,acc,count_time_gen=True)
                            
    return acc,list_scores
    

#--------------------------------------------------------------------------------------------------
# Funciones para el proceso de aprendizaje o búsqueda de la óptima distribución de molinos.
#--------------------------------------------------------------------------------------------------

# FUNCIÓN 6 (Inicializar las características del terreno y las turbinas sobre los cuales se aplicará la optimización)
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

# FUNCIÓN 7 (Evaluar el desempeño del diseño del parque eólico)
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


# FUNCIÓN 8 (Aprender la solución)
def learn(seed,heuristic,heuristic_param,maxfeval=500,popsize=50): 

    global folder_name
    folder_name='File_'+str(heuristic)+'_'+str(heuristic_param)
    os.makedirs(folder_name)

    # Inicializar el terreno y las turbinas que se desean colocar sobre el mismo.
    default_windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/')


    # Máximo tiempo de ejecución.
    global max_time
    max_time=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/max_time.npy')
    
    # Función para transformar el valor escalado de los parámetros en los valores reales.
    def transform_to_problem_dim(list_coord):
        lbound = np.zeros(default_windFLO.nTurbines*2) # Límite inferior real.
        ubound = np.ones(default_windFLO.nTurbines*2)*2000 # Límite superior real.
        return lbound + list_coord*(ubound - lbound)

    # Inicializar contadores de tiempo.
    global time_proc
    time_proc=0

    global time_acc
    time_acc=0

    # Otras inicializaciones.
    gen=0
    accuracy=1

    # Aplicar algoritmo CMA-ES para la búsqueda de la solución.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(default_windFLO.nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfeval, 'popsize':popsize})
    
    while time_proc+time_acc<max_time:

        # Construir generación.
        solutions = es.ask()

        # Transformar los valores escalados de los parámetros a los valores reales.
        real_solutions=[transform_to_problem_dim(list_coord) for list_coord in solutions]

        # Reajustar el accuracy según el heurístico seleccionado.
        if gen==0:
            accuracy,list_scores=execute_heuristic(gen,accuracy,real_solutions,seed,[],heuristic,heuristic_param)

        else:
            df_seed=pd.DataFrame(df)
            df_seed=df_seed[df_seed[1]==seed]
            accuracy,list_scores=execute_heuristic(gen,accuracy,real_solutions,seed,list(df_seed[5]),heuristic,heuristic_param)

        # Para construir la siguiente generación.
        es.tell(solutions,list_scores)

        # Acumular datos de interés.
        score = EvaluateFarm(transform_to_problem_dim(es.result.xbest),default_windFLO)
        df.append([heuristic_param,seed,gen,-score,accuracy,np.var(list_scores),time_proc,time_acc,time_acc+time_proc])

        gen+=1

    os.rmdir(folder_name)

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Lista de semillas de entrenamiento.
list_seeds=range(1,51,1)

# Preparar lista de argumentos.
# sample_size_freq='BisectionAndPopulation'
sample_size_freq='BisectionOnly'
list_arg=[['II',5],['II',10]]#['I',0.8],['I',0.95],

# Construir bases de datos.
for arg in tqdm(list_arg):
    
    df=[]

    heuristic=arg[0]
    heuristic_param=arg[1]

    for seed in tqdm(list_seeds):
        learn(seed,heuristic,heuristic_param,maxfeval=500,popsize=50)
    
    df=pd.DataFrame(df,columns=['heuristic_param','seed','n_gen','score','accuracy','variance','elapsed_time_proc','elapsed_time_acc','elapsed_time'])
    df.to_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(heuristic_param)+'.csv')

# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')

# Juntar bases de datos.
def concat_same_heuristic_df(list_arg):
    heuristic_param_dict={}
    for arg in list_arg:
        heuristic=arg[0]
        parameter=arg[1]
        if heuristic not in heuristic_param_dict:
            heuristic_param_dict[heuristic]=[parameter]
        else:
            heuristic_param_dict[heuristic].append(parameter)

    dict_keys=list(heuristic_param_dict.keys())

    for key in dict_keys:
        list_param=heuristic_param_dict[key]
        first=True
        for param in list_param:
            
            if first:
                df=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv')
                first=False
            else:
                df_new=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv')
                df=pd.concat([df,df_new],ignore_index=True)

        df.to_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h'+str(key)+'_'+str(sample_size_freq)+'.csv')

concat_same_heuristic_df(list_arg)
