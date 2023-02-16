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


sys.path.append('WindFLO/API')
from WindFLO import WindFLO

#==================================================================================================
# FUNCIONES
#==================================================================================================

# Función para inicializar las características del terreno y las turbinas sobre los cuales se aplicará la optimización.
def get_windFLO_with_accuracy(accuracy=1):

    # Configuración y parámetros de WindFLO.
    windFLO = WindFLO(
    inputFile = 'WindFLO/Examples/Example1/WindFLO.dat', # Archivo input para leer.
    libDir = 'WindFLO/release/', # Ruta a la librería compartida libWindFLO.so.
    turbineFile = 'WindFLO/Examples/Example1/V90-3MW.dat',# Parámetros de las turbinas.
    terrainfile = 'WindFLO/Examples/Example1/terrain.dat', # Archivo del terreno.
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

    return windFLO.farmPower

# Generar solución aleatoria.
def generate_random_solution(seed,windFLO):

    # Función para tranformar solución al
    def transform_to_problem_dim(x):
        lbound = np.zeros(windFLO.nTurbines*2)    
        ubound = np.ones(windFLO.nTurbines*2)*2000
        return lbound + x*(ubound - lbound)

    # Generar solución aleatoria.
    np.random.seed(seed)
    solution=transform_to_problem_dim(np.random.random(windFLO.nTurbines*2))
    return solution

# Para representar de forma gráfica el resultado.
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

# Seleccionar una posible solución de forma aleatoria.
solution=generate_random_solution(0,windFLO)

# Evaluar soluciones.
t=time.time()
score=EvaluateFarm(solution, windFLO)
elapsed=time.time()-t

print('Score: '+str(score)+' Time: '+str(elapsed))

# Dibujar solución.
plot_WindFLO(windFLO,'results/figures/WindFLO','EvaluationExample')