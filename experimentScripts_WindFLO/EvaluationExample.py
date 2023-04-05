# Mediante este script se evalua una posible solucion aleatoria para la distribucion de los 
# molinos en el entorno del ejemplo "example1". Se obtienen el score, el tiempo de evaluacion y
# la representacion grafica de la solucion.

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


sys.path.append('WindFLO/API')
from WindFLO import WindFLO

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCION 1 (Inicializar las caracteristicas del terreno y las turbinas sobre los cuales se aplicara la optimizacion)
def get_windFLO_with_accuracy(accuracy=1):

    # Configuracion y parametros de WindFLO.
    windFLO = WindFLO(
    inputFile = 'WindFLO/Examples/Example1/WindFLO.dat', # Archivo input para leer.
    libDir = 'WindFLO/release/', # Ruta a la libreria compartida libWindFLO.so.
    turbineFile = 'WindFLO/Examples/Example1/V90-3MW.dat',# Parametros de las turbinas.
    terrainfile = 'WindFLO/Examples/Example1/terrain.dat', # Archivo del terreno.
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

# FUNCION 3 (Generar solucion aleatoria)
def generate_random_solution(seed,windFLO):

    # Funcion para tranformar solucion al
    def transform_to_problem_dim(x):
        lbound = np.zeros(windFLO.nTurbines*2)    
        ubound = np.ones(windFLO.nTurbines*2)*2000
        return lbound + x*(ubound - lbound)

    # Generar solucion aleatoria.
    np.random.seed(seed)
    solution=transform_to_problem_dim(np.random.random(windFLO.nTurbines*2))
    return solution

# FUNCION 4 (Representar de forma grafica el resultado)
def plot_WindFLO(windFLO,path,file_name):

    # Resultado en 2D.
    fig = plt.figure(figsize=(8,5), edgecolor = 'gray', linewidth = 2)
    ax = windFLO.plotWindFLO2D(fig, plotVariable = 'P', scale = 1.0e-3, title = 'P [kW]')
    windFLO.annotatePlot(ax)
    plt.savefig(path+'/'+file_name+"_2D.pdf")

    # Resultado en 3D.
    fig = plt.figure(figsize=(8,5), edgecolor = 'gray', linewidth = 2)
    ax = windFLO.plotWindFLO3D(fig)
    windFLO.annotatePlot(ax)
    plt.savefig(path+'/'+file_name+"_3D.pdf")

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Inicializar el entorno.
windFLO=get_windFLO_with_accuracy()

# Seleccionar una posible solucion de forma aleatoria.
solution=generate_random_solution(0,windFLO)

# Evaluar soluciones.
t=time.time()
score=EvaluateFarm(solution, windFLO)
elapsed=time.time()-t

print('Score: '+str(score)+' Time: '+str(elapsed))

# Dibujar solucion.
plot_WindFLO(windFLO,'results/figures/WindFLO','EvaluationExample')

# Eliminar ficheros auxiliares.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')