'''
This script saves the relevant information extracted from the execution process of the CMA-ES 
algorithm on the Turbines environment, using 10 different N values and considering 
a total of 100 seeds for each one. 
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
import cma
import time
import os
from tqdm import tqdm
import pandas as pd
import itertools
from joblib import Parallel, delayed
import shutil

import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def transform_turb_params(scaled_x):
    '''Transform the scaled values of the parameters to the real values.'''

    # Set the ranges of the parameters defining the turbine design.
    blade_number = [3,5,7]# Blade-number gene.
    sigma_hub = [0.4, 0.7]# Hub solidity gene.
    sigma_tip = [0.4, 0.7]# Tip solidity gene.
    nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
    tip_clearance=[0,3]# Tip-clearance gene.	  
    airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

    # List with ranges.
    bounds=np.array([
    [sigma_hub[0]    , sigma_hub[1]],
    [sigma_tip[0]    , sigma_tip[1]],
    [nu[0]           , nu[1]],
    [tip_clearance[0], tip_clearance[1]],
    [0               , 26]
    ])

    # To transform the discrete parameter blade-number.
    def blade_number_transform(posible_blade_numbers,scaled_blade_number):
        discretization=np.arange(0,1+1/len(posible_blade_numbers),1/len(posible_blade_numbers))
        detection_list=discretization>scaled_blade_number
        return posible_blade_numbers[list(detection_list).index(True)-1]

    # Transformation.
    real_x = scaled_x[1:] * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    real_bladenumber= blade_number_transform(blade_number,scaled_x[0])

    return [real_bladenumber]+list(real_x[:-1])+[round(real_x[-1])]


def fitness_function(turb_params,N=100):
    '''Evaluating a turbine design.'''

    # Build dictionary of constant parameters.
    def build_constargs_dict(N):
        # Define constant parameters.
        omega = 2100# Rotational speed.
        rcas = 0.4# Casing radius.
        airfoils = ["NACA0015", "NACA0018", "NACA0021"]# Set of possible airfoils.
        polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)# Polars.
        cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]# Target dumping coefficients.
        devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]# Input sea-state standard pressure deviations.
        weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]# Input sea-state weights.
        Nmin = 1000#Max threshold rotational speeds
        Nmax = 3200#Min threshold rotational speeds

        # Construct the dictionary needed by the fitness function.
        constargs = {"N": N,
                "omega": omega,
                "rcas": rcas,
                "airfoils": airfoils,
                "polars": polars,
                "cpobjs": cpobjs,
                "devobjs": devobjs,
                "weights": weights,
                "Nmin": Nmin,
                "Nmax": Nmax,
                "Mode": "mono"}

        return constargs

    constargs=build_constargs_dict(N)

    # Create instantaneous turbine.
    os.chdir('OptimizationAlgorithms_KONFLOT')
    turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
    os.chdir('../')

    # Calculate evaluation.
    scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

    return -scores[1]


def learn(accuracy,seed,popsize=20):
    '''Run the CMA-ES algorithm with specific seed and N accuracy values.'''

    # Initialize CMA-ES.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(6), 0.33,inopts={'bounds': [0, 1],'seed':seed,'popsize':popsize})

    # Initialize time counters.
    eval_time = 0

    # Evaluate populations designs until the maximum time is exhausted.
    n_gen=0
    while eval_time<max_time:

        # New population.
        solutions = es.ask()

        # Transform the scaled values of the parameters to the real values.
        list_turb_params=[transform_turb_params(x) for x in solutions]

        # Obtain scores and times per evaluation.
        os.makedirs(sys.path[0]+'/PopulationScores'+str(task))

        def parallel_f(turb_params,index):
            score=fitness_function(turb_params, N=int(default_N*accuracy))
            np.save(sys.path[0]+'/PopulationScores'+str(task)+'/'+str(index),score)

        t=time.time()
        Parallel(n_jobs=4)(delayed(parallel_f)(list_turb_params[i],i) for i in range(popsize))
        eval_time+=time.time()-t

        def obtain_score_list(popsize):
            list_scores=[]
            for i in range(popsize):
                score=float(np.load(sys.path[0]+'/PopulationScores'+str(task)+'/'+str(i)+'.npy'))
                list_scores.append(score) 
            shutil.rmtree(sys.path[0]+'/PopulationScores'+str(task),ignore_errors=True)
            return list_scores
        list_scores=obtain_score_list(popsize)

        # To build the next generation.
        es.tell(solutions, list_scores)

        # Accumulate data of interest.
        test_score= fitness_function(transform_turb_params(es.result.xbest))
        df.append([accuracy,seed,n_gen,-test_score,eval_time])

        n_gen+=1

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Grids.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]# List of accuracies.                    
list_seeds=range(2,102,1)# List with train seeds.

# Combination of values (accuracy,seed) that define each task of the execution in the cluster.
list_tasks=list(itertools.product(list_acc,list_seeds))

# Parameters.
default_N=100
max_time=60*60 # one hour per seed and accuracy value.

# Build a database with relevant data on task execution.
for accuracy in list_acc:

    for seed in tqdm(list_seeds):
        task=seed
        df=[]
        learn(accuracy,seed)

    # Save database.
    df=pd.DataFrame(df,columns=['accuracy','seed','n_gen','score','elapsed_time'])
    df.to_csv('results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc'+str(accuracy)+'.csv')

# Save list with accuracy values.
np.save('results/data/Turbines/ConstantAccuracyAnalysis/list_acc',list_acc)

# Save runtime limit.
np.save('results/data/Turbines/ConstantAccuracyAnalysis/max_time',max_time)

