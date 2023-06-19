'''
This script evaluates a possible random solution for the distribution of the mills in the environment 
of the example "WindFLO/Examples/Example1/example1.py". The score, the evaluation time and the graphical 
representation of the solution are obtained.
'''
#==================================================================================================
# LIBRARIES
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
# FUNCTIONS
#==================================================================================================
def get_windFLO_with_accuracy(accuracy=1):
    '''Initialize the characteristics of the terrain and turbines on which the optimization will be applied.'''

    # Configuration and parameters.
    windFLO = WindFLO(
    inputFile = 'WindFLO/Examples/Example1/WindFLO.dat', # Input file to read.
    libDir = 'WindFLO/release/', # Path to the shared library libWindFLO.so.
    turbineFile = 'WindFLO/Examples/Example1/V90-3MW.dat',# Turbine parameters.
    terrainfile = 'WindFLO/Examples/Example1/terrain.dat', # File associated with the terrain.
    nTurbines = 25, # Number of turbines.

    monteCarloPts = round(1000*accuracy)# Parameter whose accuracy will be modified.
    )

    # Change the default terrain model from RBF to IDW.
    windFLO.terrainmodel = 'IDW'

    return windFLO

def EvaluateFarm(x, windFLO):
    '''Evaluating the performance of a single solution.'''
    
    k = 0
    for i in range(0, windFLO.nTurbines):
        for j in range(0, 2):
            # unroll the variable vector 'x' and assign it to turbine positions
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1

    # Run WindFLO analysis.
    windFLO.run(clean = True) 

    return windFLO.farmPower

def generate_random_solution(seed,windFLO):
    '''Generate random solution.'''

    # Function to transform solution from scaled values to real values.
    def transform_to_problem_dim(x):
        lbound = np.zeros(windFLO.nTurbines*2)    
        ubound = np.ones(windFLO.nTurbines*2)*2000
        return lbound + x*(ubound - lbound)

    # Random solution.
    np.random.seed(seed)
    solution=transform_to_problem_dim(np.random.random(windFLO.nTurbines*2))
    return solution

def plot_WindFLO(windFLO,path,file_name):
    '''Graphically represent the solution.'''

    # Results in 2D.
    fig = plt.figure(figsize=(8,5), edgecolor = 'gray', linewidth = 2)
    ax = windFLO.plotWindFLO2D(fig, plotVariable = 'P', scale = 1.0e-3, title = 'P [kW]')
    windFLO.annotatePlot(ax)
    plt.savefig(path+'/'+file_name+"_2D.pdf")

    # Results in 3D.
    fig = plt.figure(figsize=(8,5), edgecolor = 'gray', linewidth = 2)
    ax = windFLO.plotWindFLO3D(fig)
    windFLO.annotatePlot(ax)
    plt.savefig(path+'/'+file_name+"_3D.pdf")

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

# Initialize environment.
windFLO=get_windFLO_with_accuracy()

# Select a possible solution randomly.
solution=generate_random_solution(0,windFLO)

# Evaluate solution.
t=time.time()
score=EvaluateFarm(solution, windFLO)
elapsed=time.time()-t

print('Score: '+str(score)+' Time: '+str(elapsed))

# Draw solution..
plot_WindFLO(windFLO,'results/figures/WindFLO/EvaluationExample','EvaluationExample')

# Delete auxiliary files.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')