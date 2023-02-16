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
# FUNCIONES
#==================================================================================================

# Función para inicializar las características del terreno y las turbinas sobre los cuales se 
# aplicará la optimización.
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

    return windFLO.farmPower

# Generar conjunto de soluciones.
def build_solution_set(n_sample,seed):

    # Construir entorno por defecto.
    windFLO=get_windFLO_with_accuracy()

    # Función para tranformar solución al
    def transform_to_problem_dim(x):
        lbound = np.zeros(windFLO.nTurbines*2)    
        ubound = np.ones(windFLO.nTurbines*2)*2000
        return lbound + x*(ubound - lbound)

    # generar conjunto de soluciones.
    np.random.seed(seed)
    solution_set=[]
    for _ in range(n_sample):
        solution_set.append(transform_to_problem_dim(np.random.random(windFLO.nTurbines*2)))

    return solution_set

# Función que evaluar un conjunto de soluciones.
def evaluate_solution_set(solution_set,accuracy):

    # Crear carpeta auxiliar para guardar en cada ejecución en paralelo sus propios archivos 
    # auxiliares, y no se mezclen con los de las demás ejecuciones.
    folder_name='File'+str(accuracy)
    os.makedirs(folder_name)

    # Generar entorno con accuracy adecuado.
    windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)

    
    # Evaluar las soluciones e ir guardando la información relevante.
    for i in tqdm(range(len(solution_set))):
        # Evaluación.
        t=time.time()
        score=EvaluateFarm(solution_set[i], windFLO)
        elapsed=time.time()-t

        # Guardar información.
        df.append([accuracy,i+1,score,elapsed])

    # Borrar carpeta auxiliar.
    os.rmdir(folder_name)


#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Construir conjunto de 100 posibles soluciones.
solution_set=build_solution_set(100,0)

# Lista de accuracys a considerar (para que el código funcione lo elementos de esta lista no deben superar 
# el número de cores del ordenador, en caso de querer evaluar más accuracys hacerlo en una nueva ejecución).
list_acc=[1.0,0.5,0.2,0.1,0.05,0.02,0.01,0.005] 

# Inicializar base de datos donde se guardará la información.
df=[]

# Función para realizar la ejecución en paralelo.
def parallel_processing(arg):

    # Evaluar conjunto de puntos.
    evaluate_solution_set(solution_set,arg)

    # Guardar base de datos.
    global df
    df=pd.DataFrame(df,columns=['accuracy','n_solution','score','time'])
    df.to_csv('results/data/WindFLO/df_UnderstandingAccuracy'+str(arg)+'.csv')

# Procesamiento en paralelo.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,list_acc)
pool.close()

# Juntar bases de datos.
df=pd.read_csv('results/data/WindFLO/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv', index_col=0)
os.remove('results/data/WindFLO/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv')
for accuracy in list_acc[1:]:
    # Leer, eliminar y unir.
    df_new=pd.read_csv('results/data/WindFLO/df_UnderstandingAccuracy'+str(accuracy)+'.csv', index_col=0)
    os.remove('results/data/WindFLO/df_UnderstandingAccuracy'+str(accuracy)+'.csv')
    df=pd.concat([df,df_new],ignore_index=True)
df.to_csv('results/data/WindFLO/df_UnderstandingAccuracy.csv')

# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')

