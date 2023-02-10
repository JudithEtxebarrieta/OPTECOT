'''

    Example 1 on using WindFLO library

  Purpose:
      
      This example demonstrate how to use the Python WindFLO API for serial optimization 
      of wind farm layout. It uses WindFLO to analyze each layout configuration
      and a the particle swarm optimization (PSO) algorithm from the pyswarm package
      to perform the optimization.  The layout is optimized for maximum power generation
      and incorporates constraints on the minimum allowable clearance between turbines. 
  
      IMPORTANT: The PSO minimizes the fitness function, therefore the negative of the
                 farm power is returned by the EvaluateFarm function
  
  Licensing:
  
    This code is distributed under the Apache License 2.0 
    
  Author:
      Sohail R. Reddy
      sredd001@fiu.edu
      
'''

# En este caso consideramos el parametro que nos pone el autor de monteCarloPts = 1000
# Lo mejor seria medir el tiempo

# El stopping criterion creo que tiene que ser el tiempo que tarda el algoritmo de media cuando se ejecuta con accuracy 1.
# Por defecto el algoritmo de busqueda que venia era pso pero no me gusta nada este algoritmo y ademas
# generaba muchas soulciones fuera de los valores aceptables. Por eso le he puesto CMA-ES con el mismo numero de
# generaciones y tamaño de poblacion.
maxfevals = 500
popsize = 50




import sys
# Append the path to the API directory where WindFLO.py is
sys.path.append('API/')
# Append the path to the Optimizers directory where pso.py is
sys.path.append('Optimizers/')


import time
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




import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

from pso import pso
from WindFLO import WindFLO
import cma

###############################################################################
#    WindFLO Settings and Params
nTurbines = 25            # Number of turbines
libPath = 'release/'        # Path to the shared library libWindFLO.so
inputFile = 'Examples/Example1/WindFLO.dat'    # Input file to read
turbineFile = 'Examples/Example1/V90-3MW.dat'    # Turbine parameters
terrainfile = 'Examples/Example1/terrain.dat'    # Terrain file
diameter = 90.0            # Diameter to compute clearance constraint






def get_windFLO_with_accuracy(accuracy):
    windFLO = WindFLO(
    inputFile = 'Examples/Example1/WindFLO.dat',
    nTurbines = nTurbines,
    libDir = libPath,
    turbineFile = turbineFile,
    terrainfile = terrainfile,
    monteCarloPts = round(1000*accuracy),
    octreeMaxPts = 10,
    )
    windFLO.terrainmodel = 'IDW'    # change the default terrain model from RBF to IDW
    return windFLO



# Function to evaluate the farm's performance
def EvaluateFarm(x, windFLO):

    k = 0
    for i in range(0, nTurbines):
        for j in range(0, 2):
               # unroll the variable vector 'x' and assign it to turbine positions
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1
    # Run WindFLO analysis

    windFLO.run(clean = True)    
    
    # Return the farm power or any other farm output
      # NOTE: The negative value is returns since PSO minimizes the fitness value
    return -windFLO.farmPower






# Main function
def main(seed, accuracy):
    
    # Two variable per turbines (its x and y coordinates)
    lbound = np.zeros(nTurbines*2)    #lower bounds of x and y of turbines
    ubound = np.ones(nTurbines*2)*2000    #upper bounds of x and y of turbines

    def transform_to_problem_dim(x):
        return lbound + x*(ubound - lbound)


    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfevals, 'popsize':popsize})
    
    sw = stopwatch()
    
    windFLO = get_windFLO_with_accuracy(accuracy)

    from tqdm import tqdm as tqdm
    for i in tqdm(range(round(maxfevals / popsize))):
        solutions = es.ask()
        es.tell(solutions, [EvaluateFarm(transform_to_problem_dim(x), windFLO) for x in solutions])
        # es.logger.add()  # write data to disc to be plotted
        # print("---------")
        # print("Funcion objetivo: ", es.result.fbest)
        # print("Mejor solucion so far: ", transform_to_problem_dim(es.result.xbest))
        # print("Evaluaciones funcion objetivo: ", es.result.evaluations)
        # # # print("Tiempo: ", sw.get_time())
        # print("---------")
    es.result_pretty()


    xBest = transform_to_problem_dim(es.result.xbest)
    bestPower = es.result.fbest

    print(bestPower, sw.get_time(), sep=",")

    windFLO.run(clean = True, resFile = 'WindFLO.res')
    
    # Plot the optimum configuration    
    fig = plt.figure(figsize=(8,5), edgecolor = 'gray', linewidth = 2)
    ax = windFLO.plotWindFLO2D(fig, plotVariable = 'P', scale = 1.0e-3, title = 'P [kW]')
    windFLO.annotatePlot(ax)
    plt.savefig("result_fig.pdf")
    
    # save the optimum to a file
    np.savetxt('optimum.dat', xBest)

