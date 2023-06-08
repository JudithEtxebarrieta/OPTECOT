'''
This script stores in a database the relevant information associated with the evaluation of a set of 
turbine designs. The set is formed by a total of 10 or 100 turbine designs chosen randomly, and all
the designs of the set are evaluated considering different values of N. A database is built 
with the information of scores and execution times per evaluation.
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

import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")
import turbine_classes
import MathTools as mt

from tqdm import tqdm
import pandas as pd

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def choose_random_configurations(list_seeds,blade_number=[3,5,7]):
	'''Create a random set of turbine designs (as many designs as the size of the list of seeds).'''

	# Define ranges of parameters defining the turbine design.
	blade_number = blade_number# Blade-number gene.
	sigma_hub = [0.4, 0.7]# Hub solidity gene.
	sigma_tip = [0.4, 0.7]# Tip solidity gene.
	nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
	tip_clearance=[0,3]# Tip-clearance gene.	  
	airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

	# Save as many random configurations as seeds in list_seeds.
	list_turb_params=[]
	for seed in list_seeds: 
		np.random.seed(seed)
		if type(blade_number)==list:
			select_blade_number=np.random.choice(blade_number)
		if type(blade_number)==int:
			select_blade_number=blade_number
		turb_params = [select_blade_number,
		       np.random.uniform(sigma_hub[0], sigma_hub[1]),
		       np.random.uniform(sigma_tip[0], sigma_tip[1]),
		       np.random.uniform(nu[0], nu[1]),
		       np.random.uniform(tip_clearance[0], tip_clearance[1]),
		       np.random.choice(airfoil_dist)]
		list_turb_params.append(turb_params)

	return list_turb_params

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

def evaluation(turb_params,N=100):

	'''Evaluating a turbine design.'''

	# Build dictionary of constant parameters.
	constargs=build_constargs_dict(N)

	# Create instantaneous turbine.
	os.chdir('OptimizationAlgorithms_KONFLOT')
	turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
	os.chdir('../')

	# Evaluate design.
	scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

	total_fitness=scores[1]
	partial_fitness=[]
	for e, score in enumerate(scores[0]):
		partial_fitness.append(score)

	return total_fitness,partial_fitness

def set_evaluation(set_turb_params,N,accuracy=None):
	'''Evaluate a set of turbine designs.'''

	# Calculate scores and run times for the evaluation of each design.
	all_scores=[]
	all_times=[]
	n_solution=0
	for turb_param in tqdm(set_turb_params):

		t=time.time()
		total_fitness,partial_fitness=evaluation(turb_param,N=N)
		elapsed=time.time()-t
		n_solution+=1

		all_scores.append(total_fitness)
		all_times.append(elapsed)
		if accuracy!=None:
			df.append([accuracy,n_solution,total_fitness,elapsed])

	# Calculate data to be stored in the database.
	ranking=from_argsort_to_ranking(np.argsort(all_scores))
	total_time=sum(all_times)
	time_per_eval=np.mean(all_times)

	if accuracy==None:
		return [N,all_scores,ranking,all_times,total_time,time_per_eval]


def from_argsort_to_ranking(list):
    '''Obtain ranking from a list get after applying "np.argsort" on an original list.'''
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

#--------------------------------------------------------------------------------------------------
# First motivational analysis.
#--------------------------------------------------------------------------------------------------
# Randomly choose a set of 10 configurations.
list_seeds=range(0,10)# Seed setting for reproducibility.
set_turb_params = choose_random_configurations(list_seeds)

# Grid for N.
grid_N=[1000,900,800,700,600,500,400,300,200,100,50,45,40,35,30,25,20,15,10,5]

# Store in a database the data associated with the evaluation of the set of designs for each value of N considered.
df=[]
for n in tqdm(grid_N):
	df.append(set_evaluation(set_turb_params,n))

df=pd.DataFrame(df,columns=['N','all_scores','ranking','all_times','total_time','time_per_eval'])
df.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyI.csv')

#--------------------------------------------------------------------------------------------------
# Second motivational analysis.
#--------------------------------------------------------------------------------------------------

# Analysis to select the blade-number definition to be considered.
for blade_number in [3,5,7,[3,5,7]]:
	# Randomly choose a set of 100 configurations.
	list_seeds=range(0,100)# Seed setting for reproducibility.
	set_turb_params = choose_random_configurations(list_seeds,blade_number=blade_number)

	# Grid for N.
	list_n=range(5,101,1)
	list_acc=[]
	for n in list_n:
		list_acc.append(n/100)
	list_acc=[1]

	# Store in a database the data associated with the evaluation of the set of designs for each value of N considered.
	default_N=100
	df=[]
	for acc in tqdm(list_acc):
		set_evaluation(set_turb_params,int(default_N*acc),accuracy=acc)

	df_motivation=pd.DataFrame(df,columns=['accuracy','n_solution','score','time'])
	if type(blade_number)==list:
		df_motivation.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumberAll.csv')
	else:
		df_motivation.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber'+str(blade_number)+'.csv')

# Reduce database of the selected blade-number (list of all possible) to the accuracy values of interest.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
df=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumberAll.csv',index_col=0)
df=df.iloc[[i in list_acc for i in df['accuracy']]]
df.to_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII.csv')

#--------------------------------------------------------------------------------------------------
# For the definition of the values (times) on which the bisection will be applied.
#--------------------------------------------------------------------------------------------------
# Save database.
df.columns=['accuracy','n_solution','score','cost_per_eval']
df_bisection=df[['accuracy','cost_per_eval']]
df_bisection=df_bisection.groupby('accuracy').mean()
df_bisection.to_csv('results/data/Turbines/UnderstandingAccuracy/df_Bisection.csv')




