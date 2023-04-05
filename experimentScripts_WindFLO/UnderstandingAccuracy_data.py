# Mediante este script se evaluan 100 soluciones aleatorias considerando 10 valores de accuracy
# diferentes para el parametro monteCarloPts. Los datos relevantes (scores y tiempos de ejecucion
# por evaluacion) se almacenan para despues poder acceder a ellos.

#==================================================================================================
# LIBRERIAS
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

# FUNCION 1 (Inicializar las caracteristicas del terreno y las turbinas sobre los cuales se aplicara la optimizacion)
def get_windFLO_with_accuracy(momentary_folder='',accuracy=1):

    # Configuracion y parametros de WindFLO.
    windFLO = WindFLO(
    inputFile = 'WindFLO/Examples/Example1/WindFLO.dat', # Archivo input para leer.
    libDir = 'WindFLO/release/', # Ruta a la libreria compartida libWindFLO.so.
    turbineFile = 'WindFLO/Examples/Example1/V90-3MW.dat',# Parametros de las turbinas.
    terrainfile = 'WindFLO/Examples/Example1/terrain.dat', # Archivo del terreno.
    runDir=momentary_folder,
    nTurbines = 25, # Numero de turbinas.

    monteCarloPts = round(1000*accuracy)# Parametro del cual se modificara su precision.
    )

    # Cambiar el modelo de terreno predeterminado de RBF a IDW.
    windFLO.terrainmodel = 'IDW'

    return windFLO

# FUNCION 2 (Evaluar el desempeño del diseño del parque eolico)
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

# FUNCION 3 (Generar conjunto de soluciones)
def build_solution_set(n_sample,seed):

    # Construir entorno por defecto.
    windFLO=get_windFLO_with_accuracy()

    # Funcion para tranformar solucion al
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

# FUNCION 4 (Evaluar un conjunto de soluciones)
def evaluate_solution_set(solution_set,accuracy):

    # Crear carpeta auxiliar para guardar en cada ejecucion en paralelo sus propios archivos 
    # auxiliares, y no se mezclen con los de las demas ejecuciones.
    folder_name='File'+str(accuracy)
    os.makedirs(folder_name)

    # Generar entorno con accuracy adecuado.
    windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)

    
    # Evaluar las soluciones e ir guardando la informacion relevante.
    for i in tqdm(range(len(solution_set))):
        # Evaluacion.
        t=time.time()
        score=EvaluateFarm(solution_set[i], windFLO)
        elapsed=time.time()-t

        # Guardar informacion.
        df.append([accuracy,i+1,score,elapsed])

    # Borrar carpeta auxiliar.
    os.rmdir(folder_name)


#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Construir conjunto de 100 posibles soluciones.
solution_set=build_solution_set(100,0)

#--------------------------------------------------------------------------------------------------
# Para el analisis de motivacion.
#--------------------------------------------------------------------------------------------------
# Lista de accuracys a considerar (valores equidistantes para despues facilitar la interpolacion).
list_acc=np.arange(0.001,1.0+(1.0-0.001)/9,(1.0-0.001)/9)

# Guardar datos de scores y tiempos por evaluacion usando diferentes valores de accuracy.
for accuracy in list_acc:

    # Inicializar base de datos donde se guardara la informacion.
    df=[]

    # Evaluar conjunto de puntos.
    evaluate_solution_set(solution_set,accuracy)

    # Guardar base de datos.
    df=pd.DataFrame(df,columns=['accuracy','n_solution','score','time'])
    df.to_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(accuracy)+'.csv')

# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')

# Juntar bases de datos.
df=pd.read_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv', index_col=0)
os.remove('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv')
for accuracy in list_acc[1:]:
    # Leer, eliminar y unir.
    df_new=pd.read_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(accuracy)+'.csv', index_col=0)
    os.remove('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(accuracy)+'.csv')
    df=pd.concat([df,df_new],ignore_index=True)
df.to_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy.csv')

#--------------------------------------------------------------------------------------------------
# Para la definicion de los valores (tiempo) sobre los cuales se aplicara la biseccion.
#-------------------------------------------------------------------------------------------------- 

# Guardar base de datos.
df_bisection=df.rename(columns={'time':'cost_per_eval'})
df_bisection=df_bisection[['accuracy','cost_per_eval']]
df_bisection=df_bisection.groupby('accuracy').mean()
df_bisection.to_csv('results/data/WindFLO/UnderstandingAccuracy/df_Bisection.csv')
