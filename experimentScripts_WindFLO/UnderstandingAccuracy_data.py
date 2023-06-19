'''
This script evaluates 100 random solutions considering 10 different accuracy values for the 
parameter monteCarloPts. The relevant data (scores and execution times per evaluation) are 
stored for later access.

Based on: https://github.com/sohailrreddy/WindFLO
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
import concurrent.futures as cf
import psutil as ps


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
def build_solution_set(n_sample,seed):
    '''Generate set of solutions.'''

    # Build default environment.
    windFLO=get_windFLO_with_accuracy()

    # Function to transform solution from scaled values to real values.
    def transform_to_problem_dim(x):
        lbound = np.zeros(windFLO.nTurbines*2)    
        ubound = np.ones(windFLO.nTurbines*2)*2000
        return lbound + x*(ubound - lbound)

    # Generate a set of solutions.
    np.random.seed(seed)
    solution_set=[]
    for _ in range(n_sample):
        solution_set.append(transform_to_problem_dim(np.random.random(windFLO.nTurbines*2)))

    return solution_set

def EvaluateFarm(x, windFLO):
    '''Evaluating the performance of a single solution.'''
    
    k = 0
    for i in range(0, windFLO.nTurbines):
        for j in range(0, 2):
            # Unroll the variable vector 'x' and assign it to turbine positions.
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1

    # Run WindFLO analysis.
    windFLO.run(clean = True) 

    return windFLO.farmPower

def evaluate_solution_set(solution_set,accuracy):
    '''Evaluate a set of solutions given the accuracy of the monteCarloPts parameter.'''

    # Create auxiliary folder to save in each parallel execution its own auxiliary files, so that 
    # they are not mixed with those of the other executions.
    folder_name='File'+str(accuracy)
    os.makedirs(folder_name)

    # Generate an environment with indicated accuracy.
    windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)

    # Evaluate solutions and save relevant information.
    for i in tqdm(range(len(solution_set))):
        # Evaluation.
        t=time.time()
        score=EvaluateFarm(solution_set[i], windFLO)
        elapsed=time.time()-t

        # save information.
        df.append([accuracy,i+1,score,elapsed])

    # Delete auxiliary folder.
    os.rmdir(folder_name)

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Construct set of 100 possible solutions.
solution_set=build_solution_set(100,0)

#--------------------------------------------------------------------------------------------------
# For motivation analysis.
#--------------------------------------------------------------------------------------------------
# List of accuracys to be considered (equidistant values to facilitate interpolation).
list_acc=np.arange(0.001,1.0+(1.0-0.001)/9,(1.0-0.001)/9)

# Save score and time data per evaluation using different accuracy values.
for accuracy in list_acc:

    # Initialize the database where the information will be stored.
    df=[]

    # Evaluate set of points.
    evaluate_solution_set(solution_set,accuracy)

    # Save database.
    df=pd.DataFrame(df,columns=['accuracy','n_solution','score','time'])
    df.to_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(accuracy)+'.csv')

# Delete auxiliary files.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')

# Join databases.
df=pd.read_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv', index_col=0)
os.remove('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(list_acc[0])+'.csv')
for accuracy in list_acc[1:]:
    # Read, delete and join.
    df_new=pd.read_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(accuracy)+'.csv', index_col=0)
    os.remove('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy'+str(accuracy)+'.csv')
    df=pd.concat([df,df_new],ignore_index=True)
df.to_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy.csv')

#--------------------------------------------------------------------------------------------------
# For the definition of the values (time) on which the bisection will be applied.
#-------------------------------------------------------------------------------------------------- 
# Save database.
df_bisection=df.rename(columns={'time':'cost_per_eval'})
df_bisection=df_bisection[['accuracy','cost_per_eval']]
df_bisection=df_bisection.groupby('accuracy').mean()
df_bisection.to_csv('results/data/WindFLO/UnderstandingAccuracy/df_Bisection.csv')
