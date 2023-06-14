
'''
In this script the CMA-ES algorithm is executed using the maximum value of accuracy for the parameter
N and repeating the procedure with 5 different seeds. Although the next generations are selected according
to the evaluations obtained with the maximum accuracy, the individuals of each generation are evaluated 
with 4 additional accuracy values (besides the maximum). The data of design parameters, score and evaluation 
time associated with each solution forming each population are stored in a database for each seed considered.
'''
#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
import matplotlib as mpl
import scipy as sc
import matplotlib.pyplot as plt
import scipy.integrate as scint
import time
import os
from scipy.interpolate import interp1d
from scipy.interpolate import LSQUnivariateSpline as lsqus
from scipy.integrate import simpson
from operator import itemgetter
import openpyxl
import pickle
import copy
from functools import partial
from scipy.interpolate import interp1d
from typing import Union
import cma

import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt
import time

from cma.utilities import utils 
import pandas as pd
from tqdm import tqdm

#==================================================================================================
# CLASS
#==================================================================================================

class stopwatch:
    '''Defines the methods necessary to measure time during execution process.'''
    
    def __init__(self):
        self.reset()

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

#==================================================================================================
# FUNCTION
#==================================================================================================
def define_bounds():
    '''Define ranges of parameters defining the turbine design.'''
    sigma_hub = [0.4, 0.7]# Hub solidity gene.
    sigma_tip = [0.4, 0.7]# Tip solidity gene.
    nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
    tip_clearance=[0,3]# Tip-clearance gene.	  
    airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

    bounds=np.array([
    [sigma_hub[0]    , sigma_hub[1]],
    [sigma_tip[0]    , sigma_tip[1]],
    [nu[0]           , nu[1]],
    [tip_clearance[0], tip_clearance[1]],
    [0               , 26]
    ])

    return bounds

def build_constargs_dict(N):
    '''Build a dictionary with all the constant parameters needed to make an evaluation.'''

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

def fitness_function(turb_params,N=50):
    '''Evaluating a turbine design.'''

    # Build dictionary of constant parameters.
    constargs=build_constargs_dict(N)

    # Create instantaneous turbine.
    os.chdir('OptimizationAlgorithms_KONFLOT')
    turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
    os.chdir('../')

    # Calculate evaluation.
    if N==default_N:
        sw_stop.resume()
    global sw_eval_time
    sw_eval_time=stopwatch()
    scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')
    sw_eval_time.pause()

    if N==default_N:
        sw_stop.pause()

    return -scores[1] 

def scale_x(x,bounds):
    '''Transform the scaled values of the parameters except blade-number to the real values.'''
    return x * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

def transform_turb_params(x, blade_number,bounds):
    '''Transform the scaled values of all parameters to the real values.'''
    scaled_x = scale_x(x,bounds)
    return [blade_number]+list(scaled_x[:-1])+[round(scaled_x[-1])]


def evaluate(blade_number,bounds,seed,popsize):

    '''Run the CMA-ES algorithm with specific seed and blade-number value.'''

    # Initialize CMA-ES.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(5), 0.33,inopts={'bounds': [0, 1],'seed':seed,'popsize':popsize})

    # Initialize time counters.
    global sw_stop
    sw_stop = stopwatch()
    sw_stop.pause()

    # Evaluate the generation designs with different accuracys for N until the maximum time defined for the maximum accuracy is exhausted.
    list_turb_params=[]
    n_gen=0

    # Initialize number of evaluations counter.
    global stop_n_eval
    stop_n_eval=0

    while not es.stop():

        # New generation/population.
        n_gen+=1
        solutions = es.ask()

        # Initialize number of evaluations.
        n_eval=0

        # Evaluate new solutions and save evaluation times.
        new_scores=[]
        for x in solutions:
            # Count a new evaluation.
            n_eval+=1

            # Evaluate design with different accuracys for N.
            for accuracy in list_acc:

                # Update the N parameter.
                N=int(default_N*accuracy)

                # Parameter transformation.
                turb_params=transform_turb_params(x, blade_number,bounds)

                # Calculate score.
                new_score=fitness_function(turb_params, N)

                if N==default_N:
                    new_scores.append(new_score)
                    list_turb_params.append(turb_params)

                # Add new data to the database.
                df.append([N,seed,n_gen,n_eval,-new_score,sw_eval_time.get_time()])

        # Update number of evaluations.
        stop_n_eval+=popsize

        # Pass the values of the objective function obtained with maximum accuracy to prepare for the next iteration.
        es.tell(solutions, new_scores)
        es.logger.add()  

        # Print the current status variables on a single line.
        es.disp()

    # Save evaluated turbine designs.
    df_turb_params=pd.DataFrame(list_turb_params)
    df_turb_params.to_csv('results/data/Turbines/PopulationInfluence/df_turb_params_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv')


def new_stop_time(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
    '''Redefine the stopping criterion of the CMA-ES algorithm depending on the runtime.'''
    stop={}
    if sw_stop.get_time()>max_time:
        stop={'TIME RUN OUT':max_time}
    return stop

def new_stop_n_eval(self, check=True, ignore_list=(), check_in_same_iteration=False,
             get_value=None):
    '''Redefine the stopping criterion of the CMA-ES algorithm depending on number of evaluations.'''
    stop={}
    if stop_n_eval>max_time:
        stop={'TIME RUN OUT':max_time}
    return stop


#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# To use the new stop function.
# cma.CMAEvolutionStrategy.stop=new_stop_time
cma.CMAEvolutionStrategy.stop=new_stop_n_eval

# Define list of ranges of the parameters to be optimized (blase_number individually).
bounds=define_bounds()
list_blade_number = [3, 5, 7]# Blade-number gene.

# Grids and parameters.
list_seeds=[1,2,3,4,5]
list_acc=[1.0,0.8,0.6,0.4,0.2]
max_time=100*5 # Maximum number of evaluations.
default_N=50
popsize=10

#--------------------------------------------------------------------------------------------------
# BLADE-NUMBER=3
#--------------------------------------------------------------------------------------------------
# Set blade-number.
blade_number = 3

# Store data associated with each seed in a database.	
for seed in list_seeds:
    # Initialize database.
    df=[]

    # Evaluation.
    evaluate(blade_number,bounds,seed,popsize)

    # Save accumulated data.
    df=pd.DataFrame(df,columns=['N','seed','n_gen','n_eval','score','time'])
    df.to_csv('results/data/Turbines/PopulationInfluence/df_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv')

# Save list of seeds.
np.save('results/data/Turbines/PopulationInfluence/list_seeds',list_seeds)

# Save population size.
np.save('results/data/Turbines/PopulationInfluence/popsize',popsize)
		
		





