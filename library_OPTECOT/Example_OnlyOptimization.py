''' Solving Turbines problem using `OPTECOT.py` library. '''
#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
from OPTECOT import OPTECOT, ExperimentalGraphs

import os
import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")
import turbine_classes
import MathTools as mt

#==================================================================================================
# DEFINING REQUIREMENTS
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Compulsory parameters
#-------------------------------------------------------------------------------------------------- 
theta1=100 
theta0=10 
xdim=6 
xbounds=[[3,5,7],# Blade-number gene.
        [0.4, 0.7],# Hub solidity gene.
        [0.4, 0.7], # Tip solidity gene.
        [0.4, 0.75], # Hub-to-tip-ratio gene.
        [0,3],# Tip-clearance gene.	  
        list(np.arange(0,27)) # Airfoil dist. gene.
        ]
max_time=5*60 # 5 minutes in seconds.
objective_min=False

#--------------------------------------------------------------------------------------------------
# Implementation of the objective function.
#--------------------------------------------------------------------------------------------------
def fitness_function(turb_params,theta=100):

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

    constargs=build_constargs_dict(N=theta)

    # Create instantaneous turbine.
    os.chdir('OptimizationAlgorithms_KONFLOT')
    turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
    os.chdir('../')

    # Calculate evaluation.
    scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

    return scores[1]

#==================================================================================================
# EXAMPLE OF USE OF THE LIBRARY (to solve the optimization problem)
#==================================================================================================

# Initialize OPTECOT class.
optecot=OPTECOT(xdim=xdim,
                xbounds=xbounds,
                max_time=max_time, 
                theta0=theta0,
                theta1=theta1,
                objective_min=objective_min,
                objective_function=fitness_function
                )

# Solve problem with CMA-ES applying OPTECOT.
best_solution,score=optecot.execute_CMAES_with_OPTECOT()
print('Best solution: ',best_solution, '\nObjective value: ',score)












