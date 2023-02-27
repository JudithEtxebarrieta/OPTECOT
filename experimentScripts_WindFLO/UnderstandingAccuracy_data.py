# Mediante este script se evalúan 100 soluciones aleatorias considerando 10 valores de accuracy
# diferentes para el parámetro monteCarloPts. Los datos relevantes (scores y tiempos de ejecución
# por evaluación) se almacenan para después poder acceder a ellos.

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
import concurrent.futures as cf
import psutil as ps


sys.path.append('WindFLO/API')
from WindFLO import WindFLO

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

    return windFLO.farmPower

# FUNCIÓN 3 (Generar conjunto de soluciones)
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

# FUNCIÓN 4 (Evaluar un conjunto de soluciones)
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

# Lista de accuracys a considerar.
list_acc=[1.0,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001] 

# Guardar datos de scores y tiempos por evaluación usando diferentes valores de accuracy.
for accuracy in list_acc:

    # Inicializar base de datos donde se guardará la información.
    df=[]

    # Evaluar conjunto de puntos.
    evaluate_solution_set(solution_set,accuracy)

    # Guardar base de datos.
    df=pd.DataFrame(df,columns=['accuracy','n_solution','score','time'])
    df.to_csv('results/data/WindFLO/df_UnderstandingAccuracy'+str(accuracy)+'.csv')

# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')

# Juntar bases de datos.
df=pd.read_csv('results/data/WindFLO/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv', index_col=0)
os.remove('results/data/WindFLO/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv')
for accuracy in list_acc[1:]:
    # Leer, eliminar y unir.
    df_new=pd.read_csv('results/data/WindFLO/df_UnderstandingAccuracy'+str(accuracy)+'.csv', index_col=0)
    os.remove('results/data/WindFLO/df_UnderstandingAccuracy'+str(accuracy)+'.csv')
    df=pd.concat([df,df_new],ignore_index=True)
df.to_csv('results/data/WindFLO/df_UnderstandingAccuracy.csv')

