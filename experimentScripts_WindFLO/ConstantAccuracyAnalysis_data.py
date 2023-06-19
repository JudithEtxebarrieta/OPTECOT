'''
This script applies the CMA-ES algorithm on the WindFLO environment for a maximum runtime, 
considering 10 values for the parameter monteCarloPts of different accuracies. For each accuracy, 
the experiment is repeated using 100 different seeds. To set the running time, the algorithm is 
first run for the maximum accuracy (default situation), so the training time limit is defined as 
the minimum time between the total times needed for each seed. Finally, a database is built with 
the relevant information during execution associated with each accuracy.
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
import pandas as pd
import multiprocessing as mp

sys.path.append('WindFLO/API')
from WindFLO import WindFLO

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def get_windFLO_with_accuracy(momentary_folder='',accuracy=1):
    '''Initialize the characteristics of the terrain and turbines on which the optimization will be applied.'''

    # Configuration and parameters.
    windFLO = WindFLO(
    inputFile = 'WindFLO/Examples/Example1/WindFLO.dat', # Input file to read.
    libDir = 'WindFLO/release/', # Path to the shared library libWindFLO.so.
    turbineFile = 'WindFLO/Examples/Example1/V90-3MW.dat',# Turbine parameters.
    terrainfile = 'WindFLO/Examples/Example1/terrain.dat', # File associated with the terrain.
    runDir=momentary_folder,
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
            # unroll the variable vector 'x' and assign it to turbine positions.
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1

    # Run WindFLO analysis.
    windFLO.run(clean = True) 

    return -windFLO.farmPower

def learn(seed, accuracy,maxfeval=500,popsize=50): 
    '''Search for the optimal solution by applying the CMA-ES algorithm.'''

    global max_n_eval
    max_n_eval=maxfeval 

    # Initialize the terrain and the turbines to be placed on it.
    folder_name='File'+str(accuracy)
    os.makedirs(folder_name)
    windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)
    default_windFLO= get_windFLO_with_accuracy(momentary_folder=folder_name+'/')
    
    # Function to transform the scaled value of the parameters into the real values.
    def transform_to_problem_dim(list_coord):
        lbound = np.zeros(windFLO.nTurbines*2) # Limite inferior real.
        ubound = np.ones(windFLO.nTurbines*2)*2000 # Limite superior real.
        return lbound + list_coord*(ubound - lbound)

    # Initialize time counter.
    global eval_time
    eval_time=0

    global n_evaluations
    n_evaluations=0

    n_gen=0

    # Apply CMA-ES algorithm for solution search.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(default_windFLO.nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfeval, 'popsize':popsize})
    
    while not es.stop():

        # Build generation.
        solutions = es.ask()

        # Transform the scaled values of the parameters to the real values.
        real_solutions=[transform_to_problem_dim(list_coord) for list_coord in solutions]

        # List of scores associated with the generation.
        list_scores=[]
        for sol in real_solutions:

            t=time.time()
            fitness=EvaluateFarm(sol,windFLO)
            eval_time+=time.time()-t

            list_scores.append(fitness)
            n_evaluations+=1

        # To build the next generation.
        es.tell(solutions, list_scores)

        # Accumulate data of interest.
        score = EvaluateFarm(transform_to_problem_dim(es.result.xbest),default_windFLO)
        df_acc.append([accuracy,seed,n_gen,-score,eval_time])

        n_gen+=1
  
    os.rmdir(folder_name)

    if accuracy==1:
        return eval_time

def new_stop_max_acc(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
    '''Stopping criterion for maximum accuracy.'''
    stop={}
    if n_evaluations>max_n_eval:
        stop={'TIME RUN OUT':max_n_eval}
    return stop

def new_stop_lower_acc(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
    '''Stopping criterion for lower accuracies.'''
    stop={}
    if eval_time>max_time:
        stop={'TIME RUN OUT':max_time}
    return stop

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# List of train seeds.
list_seeds=range(1,101,1)

# List of accuracies.
list_acc=[round(i,3) for i in np.arange(1.0,0.001-(1.0-0.001)/9,-(1.0-0.001)/9)]

# Build database with relevant data for each run with an accuracy value.
for accuracy in list_acc:

    global df_acc
    df_acc=[]

    # The case of maximum accuracy must be executed first in order to define the execution time limit for the rest.
    if accuracy==1:
        cma.CMAEvolutionStrategy.stop=new_stop_max_acc
        list_total_time=[]

    # For the rest of accuracys.
    else:
        cma.CMAEvolutionStrategy.stop=new_stop_lower_acc

    for seed in tqdm(list_seeds):
        total_time=learn(seed,accuracy)
        if accuracy==1:
            list_total_time.append(total_time)

    df_acc=pd.DataFrame(df_acc,columns=['accuracy','seed','n_gen','score','elapsed_time'])
    df_acc.to_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis'+str(accuracy)+'.csv')

    if accuracy==1:
        max_time=min(list_total_time)
        np.save('results/data/WindFLO/ConstantAccuracyAnalysis/max_time',max_time)

# Save list with accuracy values.
np.save('results/data/WindFLO/ConstantAccuracyAnalysis/list_acc',list_acc)

# Delete auxiliary files.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')
