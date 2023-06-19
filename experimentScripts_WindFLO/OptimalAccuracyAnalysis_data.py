'''
In this script, the proposed heuristics are applied to the WindFLO environment. The CMA-ES 
algorithm is run on this environment using 100 different seeds for each heuristic. A database is 
built with the relevant information obtained during the execution process. 

The general descriptions of the heuristics are:
HEURISTIC I: The accuracy is updated using the constant frequency calculated in experimentScripts_general/sample_size_bisection_method.py.
HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the last population is significantly different from the previous ones.
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
import scipy as sc
import random
from itertools import combinations
import multiprocessing as mp

sys.path.append('WindFLO/API')
from WindFLO import WindFLO

#==================================================================================================
# FUNCTIONC
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Auxiliary functions to define the appropriate accuracy at each stage of the process.
#--------------------------------------------------------------------------------------------------

def spearman_corr(x,y):
    '''Calculation of Spearman's correlation between two sequences.'''
    return sc.stats.spearmanr(x,y)[0]

def from_scores_to_ranking(list_scores):
    '''Convert score list to ranking list.'''
    list_pos_ranking=np.argsort(np.array(list_scores))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

def generation_score_list(population,accuracy,count_time_acc=True,count_time_gen=False):
    '''
    Evaluate the solutions that make up a generation

    Parameters
    ==========
    population: List with the solutions that form the generation.
    accuracy: Accuracy set as optimal in the previous generation.
    count_time_acc: True or False if you want to add or not respectively, the evaluation time as additional time to adjust the accuracy.
    count_time_gen: True or False if you want to add or not to add respectively, the evaluation time as natural time for 
    the evaluation of the generation.

    Return
    ======
    List with the scores associated to each solution that forms the generation.
    '''
    global time_acc,time_proc

    # Generate an environment with indicated accuracy.
    windFLO=get_windFLO_with_accuracy(momentary_folder=folder_name+'/',accuracy=accuracy)

    # Evaluate population.
    list_scores=[]
    for sol in population:
        t=time.time()
        score=EvaluateFarm(sol,windFLO)
        elapsed_time=time.time()-t
        if count_time_acc and not count_time_gen:
            time_acc+=elapsed_time
        if count_time_gen:
            time_proc+=elapsed_time
        list_scores.append(score)

    return list_scores

#--------------------------------------------------------------------------------------------------
# Functions associated with the heuristics to be applied to adjust the accuracy.
#--------------------------------------------------------------------------------------------------
def bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_pts,threshold=0.95):
    '''
    Adapted implementation of the original bisection method.

    Parameters
    ==========
    init_acc: Initial accuracy (the minimum considered).
    population: List of solutions that form the generation.
    train_seed: Training seed.
    threshold: Threshold of the bisection method with which the interval containing the optimal accuracy value will be updated.

    Returns
    =======
    prev_m: Accuracy value selected as optimum.
    last_time_acc_increase: Evaluation time consumed in the last iteration of the bisection method.
    '''

    # Initialize lower and upper limit.
    time0=lower_time
    time1=upper_time   

    # Midpoint.
    prev_m=time0
    m=(time0+time1)/2

    # Function to calculate the correlation between the rankings of the sample_size random solutions using the current and maximum accuracy.
    def similarity_between_current_best_acc(acc,population,train_seed,first_iteration):
        global time_acc

        # Randomly select sample_size solutions that form the generation.
        random.seed(train_seed)
        ind_sol=random.sample(range(len(population)),sample_size)
        list_solutions=list(np.array(population)[ind_sol])

        # Save the scores associated with each selected solution.
        t=time.time()
        best_scores=generation_score_list(list_solutions,1,count_time_acc=first_iteration)# maximum accuracy. 
        new_scores=generation_score_list(list_solutions,acc)# New accuracy. 
        last_time_acc_increase=time.time()-t

        # Obtain rankings.
        new_ranking=from_scores_to_ranking(new_scores)# Accuracy nuevo. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximo accuracy. 
                
        # Compare the two rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)

        return metric_value,last_time_acc_increase

    # Update interval limits until the interval has a sufficiently small range.
    first_iteration=True
    stop_threshold=(time1-time0)*0.1
    while time1-time0>stop_threshold:
        metric_value,last_time_acc_increase=similarity_between_current_best_acc(np.interp(m,interpolation_pts[0],interpolation_pts[1]),population,train_seed,first_iteration)
        if metric_value>=threshold:
            time1=m
        else:
            time0=m

        prev_m=m
        m=(time0+time1)/2

        first_iteration=False
    return np.interp(prev_m,interpolation_pts[0],interpolation_pts[1]),last_time_acc_increase

def execute_heuristic(gen,acc,population,train_seed,list_accuracies,list_variances,heuristic,param):
    '''
    Execute heuristics during the training process.

    Parameters
    ==========
    gen: Generation number in the CMA-ES algorithm.
    min_acc: Minimum accuracy to be considered in the bisection method.
    acc: Accuracy associated with the previous generation.
    population: List of solutions forming the generation.
    train_seed: Training seed.
    list_accuracies: List with the optimal accuracies of the previous generations.
    list_variances: List with the variances of the scores of the previous generations.
    heuristic: Number that identifies the heuristic to be considered.
    param: Value of the parameter associated to the heuristic to be applied.

    Returns
    =======
    acc: accuracy value selected as optimal.
    time_best_acc: evaluation time consumed in the last iteration of the bisection method.
    '''

    global time_proc
    global time_acc
    global max_time
    global last_time_heuristic_accepted,heuristic_accepted
    global unused_bisection_executions, stop_heuristic

    heuristic_accepted=False

    # For the bisection method: sample size, frequency and interpolation.
    df_sample_freq=pd.read_csv('results/data/general/sample_size_freq.csv',index_col=0)
    df_interpolation=pd.read_csv('results/data/WindFLO/UnderstandingAccuracy/df_Bisection.csv')
    sample_size=int(df_sample_freq[df_sample_freq['env_name']=='WindFLO']['sample_size'])
    heuristic_freq=float(df_sample_freq[df_sample_freq['env_name']=='WindFLO']['frequency_time'])
    interpolation_acc=list(df_interpolation['accuracy'])
    interpolation_time=list(df_interpolation['cost_per_eval'])
    lower_time=min(interpolation_time)
    upper_time=max(interpolation_time)

   
    # HEURISTIC I: The accuracy is updated using a constant frequency.
    if heuristic=='I': 
        if gen==0:
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=param)
            list_scores=generation_score_list(population,acc,count_time_gen=True)
            time_acc-=time_best_acc
            last_time_heuristic_accepted=time_proc+time_acc
            heuristic_accepted=True
            
        else:
            if (time_acc+time_proc)-last_time_heuristic_accepted>=heuristic_freq:
                acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=param)
                list_scores=generation_score_list(population,acc,count_time_gen=True)
                time_acc-=time_best_acc
                last_time_heuristic_accepted=time_proc+time_acc
                heuristic_accepted=True


    # HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the last population 
    # is significantly different from the previous ones.
    if heuristic=='II': 
        if gen==0: 
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
            list_scores=generation_score_list(population,acc,count_time_gen=True)
            time_acc-=time_best_acc
            last_time_heuristic_accepted=time_proc+time_acc
            unused_bisection_executions=0
            heuristic_accepted=True
        else:
            # Check if the algorithm has started to converge.
            if len(list_accuracies)>=param[1]:
                if stop_heuristic==True:
                    heuristic_accepted==False

                if stop_heuristic==False:
                    prev_acc=list_accuracies[(-1-param[1]):-1]
                    prev_acc_high=np.array(prev_acc)>0.9
                    if sum(prev_acc_high)==param[1]:
                        stop_heuristic=True

                        acc=1
                        list_scores=generation_score_list(population,acc,count_time_gen=True)

            if len(list_variances)>=param[0]+1 and stop_heuristic==False:
                # Calculate the confidence interval.
                variance_q05=np.mean(list_variances[(-2-param[0]):-2])-2*np.std(list_variances[(-2-param[0]):-2])
                variance_q95=np.mean(list_variances[(-2-param[0]):-2])+2*np.std(list_variances[(-2-param[0]):-2])
                last_variance=list_variances[-1]
                
                # Calculate the minimum accuracy with which the maximum quality is obtained.
                if last_variance<variance_q05 or last_variance>variance_q95:

                    if (time_proc+time_acc)-last_time_heuristic_accepted>=heuristic_freq:   
                        unused_bisection_executions+=int((time_proc+time_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                        acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
                        list_scores=generation_score_list(population,acc,count_time_gen=True)
                        time_acc-=time_best_acc
                        last_time_heuristic_accepted=time_proc+time_acc
                        heuristic_accepted=True

                    else:
                        if unused_bisection_executions>0:
                            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
                            list_scores=generation_score_list(population,acc,count_time_gen=True)
                            time_acc-=time_best_acc
                            last_time_heuristic_accepted=time_proc+time_acc
                            unused_bisection_executions-=1
                            heuristic_accepted=True

    if heuristic_accepted==False:
        list_scores=generation_score_list(population,acc,count_time_gen=True)
                            
    return acc,list_scores
    
#--------------------------------------------------------------------------------------------------
# Functions for the search of the optimal solution.
#--------------------------------------------------------------------------------------------------

def get_windFLO_with_accuracy(momentary_folder='',accuracy=1):
    '''Initialize the characteristics of the terrain and turbines on which the optimization will be applied.'''

    # Configuration and parameters.
    windFLO = WindFLO(
    inputFile = 'WindFLO/Examples/Example1/WindFLO.dat', # Input file to read.
    libDir = 'WindFLO/release/', # Path to the shared library libWindFLO.so.
    turbineFile = 'WindFLO/Examples/Example1/V90-3MW.dat',# Turbine parameters.
    terrainfile = 'WindFLO/Examples/Example1/terrain.dat', # File associated with the terrain.
    runDir=momentary_folder,
    nTurbines = 25, #  Number of turbines.

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
            # Unroll the variable vector 'x' and assign it to turbine positions.
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1

    # Run WindFLO analysis.
    windFLO.run(clean = True) 

    return -windFLO.farmPower

def learn(seed,heuristic,heuristic_param,maxfeval=500,popsize=50): 
    '''Finding the optimal solution using the CMA-ES algorithm.'''
    global heuristic_accepted, stop_heuristic
    global folder_name

    # Directory where the auxiliary files that are created during the execution will be stored.
    folder_name='File_'+str(heuristic)
    os.makedirs(folder_name)

    # Initialize the terrain and the turbines to be placed on it.
    default_windFLO = get_windFLO_with_accuracy(momentary_folder=folder_name+'/')

    # Maximum execution time.
    global max_time
    max_time=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/max_time.npy')
    
    # Function to transform the scaled value of the parameters into the real values.
    def transform_to_problem_dim(list_coord):
        lbound = np.zeros(default_windFLO.nTurbines*2) # Real lower limit.
        ubound = np.ones(default_windFLO.nTurbines*2)*2000 # Real upper limit.
        return lbound + list_coord*(ubound - lbound)

    # Initialize time counters.
    global time_proc,time_acc
    time_proc=0
    time_acc=0

    # Other initializations.
    gen=0
    accuracy=1
    stop_heuristic=False

    # Apply CMA-ES algorithm for solution search.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(default_windFLO.nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfeval, 'popsize':popsize})
    
    while time_proc+time_acc<max_time:

        # Build a generation.
        solutions = es.ask()

        # Transform the scaled values of the parameters to the real values.
        real_solutions=[transform_to_problem_dim(list_coord) for list_coord in solutions]

        # Adjust the accuracy according to the selected heuristic.
        if gen==0:
            accuracy,list_scores=execute_heuristic(gen,accuracy,real_solutions,seed,[],[],heuristic,heuristic_param)

        else:
            df_seed=pd.DataFrame(df)
            df_seed=df_seed[df_seed[1]==seed]
            accuracy,list_scores=execute_heuristic(gen,accuracy,real_solutions,seed,list(df_seed[4]),list(df_seed[5]),heuristic,heuristic_param)

        # To build the next generation.
        es.tell(solutions,list_scores)

        # Accumulate data of interest.
        score = EvaluateFarm(transform_to_problem_dim(es.result.xbest),default_windFLO)
        df.append([heuristic_param,seed,gen,-score,accuracy,np.var(list_scores),heuristic_accepted,time_proc,time_acc,time_acc+time_proc])

        gen+=1

    os.rmdir(folder_name)

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

# List of training seeds.
list_seeds=range(1,101,1)

# Prepare list of arguments.
list_arg=[['I',0.8],['I',0.95],['II',[5,3]],['II',[10,3]]]

# Build database for each heuristic.
for arg in tqdm(list_arg):
    df=[]

    heuristic=arg[0]
    heuristic_param=arg[1]

    for seed in tqdm(list_seeds):
        learn(seed,heuristic,heuristic_param,maxfeval=500,popsize=50)
    
    df=pd.DataFrame(df,columns=['heuristic_param','seed','n_gen','score','accuracy','variance','update','elapsed_time_proc','elapsed_time_acc','elapsed_time'])
    df.to_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(heuristic_param)+'.csv')

# Delete auxiliary files.
os.remove(os.path.sep.join(sys.path[0].split(os.path.sep)[:-1])+'/terrain.dat')

# Join databases associated with the same heuristics considering different parameters.
def concat_same_heuristic_df(list_arg):
    heuristic_param_dict={}
    for arg in list_arg:
        heuristic=arg[0]
        parameter=arg[1]
        if heuristic not in heuristic_param_dict:
            heuristic_param_dict[heuristic]=[parameter]
        else:
            heuristic_param_dict[heuristic].append(parameter)

    dict_keys=list(heuristic_param_dict.keys())

    for key in dict_keys:
        list_param=heuristic_param_dict[key]
        first=True
        for param in list_param:
            
            if first:
                df=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv')
                first=False
            else:
                df_new=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/WindFLO/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(key)+'_param'+str(param)+'.csv')
                df=pd.concat([df,df_new],ignore_index=True)

        df.to_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h'+str(key)+'.csv')

concat_same_heuristic_df([['I',0.8],['I',0.95],['II','[5, 3]'],['II','[10, 3]']])
