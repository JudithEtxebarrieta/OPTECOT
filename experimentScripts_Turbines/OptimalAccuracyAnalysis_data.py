
'''
In this script, the proposed heuristics are applied to the Turbines environment. The CMA-ES 
algorithm is run on this environment using 100 different seeds for each heuristic. A database is built
with the relevant information obtained during the execution process. 

The general descriptions of the heuristics are:
HEURISTIC I: The accuracy is updated using the constant frequency calculated in experimentScripts_general/sample_size_bisection_method.py.
HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the last population is significantly different from the previous ones.
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
import scipy as sc
import random
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
#--------------------------------------------------------------------------------------------------
# Auxiliary functions to define the appropriate accuracy at each stage of the execution process.
#--------------------------------------------------------------------------------------------------

def spearman_corr(x,y):
    '''Calculation of Spearman's correlation between two lists.'''
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
    Generate a list of scores associated with each design in the population.

    Parameters
    ==========
    population: List of solutions forming the population.
    accuracy: Accuracy set as optimal in the previous population.
    count_time_acc: True or False if you want to add or not respectively the evaluation time as additional time to adjust the accuracy.
    count_time_gen: True or False if you want to add or not respectively the evaluation time as natural time for the population evaluation.

    Return
    ======
    list_scores: List with the scores associated to each solution that forms the population.
    '''
    global time_acc,time_proc

    # Obtain scores and times per evaluation.
    os.makedirs(sys.path[0]+'/PopulationScores'+str(task))

    def parallel_f(turb_params,index):
        score=fitness_function(turb_params, N=int(default_N*accuracy))
        np.save(sys.path[0]+'/PopulationScores'+str(task)+'/'+str(index),score)

    t=time.time()
    Parallel(n_jobs=4)(delayed(parallel_f)(population[i],i) for i in range(len(population)))
    elapsed_time=time.time()-t

    def obtain_score_list(popsize):
        list_scores=[]
        for i in range(popsize):
            score=float(np.load(sys.path[0]+'/PopulationScores'+str(task)+'/'+str(i)+'.npy'))
            list_scores.append(score) 
        shutil.rmtree(sys.path[0]+'/PopulationScores'+str(task),ignore_errors=True)
        return list_scores
    list_scores=obtain_score_list(len(population))

    # Evaluate population.
    if count_time_acc and not count_time_gen:
        time_acc+=elapsed_time
    if count_time_gen:
        time_proc+=elapsed_time

    return list_scores
#--------------------------------------------------------------------------------------------------
# Functions associated with the heuristics to be applied to adjust the accuracy.
#--------------------------------------------------------------------------------------------------

def bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_pts,threshold=0.95):
    '''Adapted implementation of bisection method.'''

    # Initialize lower and upper limit.
    time0=lower_time
    time1=upper_time   

    # Midpoint.
    prev_m=time0
    m=(time0+time1)/2

    # Function to calculate the correlation between the rankings of the sample_size random surfaces using the current and maximum accuracy..
    def similarity_between_current_best_acc(acc,population,train_seed,first_iteration):
        global time_acc

        # Randomly select sample_size surfaces forming the generation.
        random.seed(train_seed)
        ind_sol=random.sample(range(len(population)),sample_size)
        list_solutions=list(np.array(population)[ind_sol])

        # Save the scores associated with each selected solution.
        t=time.time()
        best_scores=generation_score_list(list_solutions,1,count_time_acc=first_iteration)# Maximum accuracy. 
        new_scores=generation_score_list(list_solutions,acc)# new accuracy. 
        last_time_acc_increase=time.time()-t

        # Obtain associated rankings.
        new_ranking=from_scores_to_ranking(new_scores)# New accuracy. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximum accuracy. 
                
        # Compare two rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)

        return metric_value,last_time_acc_increase

    # Reset interval limits until the interval has a sufficiently small range.
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
    Running heuristics during the training process.

    Parameters
    ==========
    gen: Generation/population number in the CMA-ES algorithm.
    acc: Accuracy associated with the previous generation.
    population: List of solutions forming the population.
    train_seed: Training seed.
    list_variances: List with the variances of the scores of the previous populations.
    param: Value of the parameter associated with the heuristic (number of previous variances to compute the confidence interval).

    Returns
    =======
    acc: Accuracy value selected as optimal.
    time_best_acc: Evaluation time consumed in the last iteration of the bisection method.
    '''

    global time_proc
    global time_acc,time_best_acc 
    global max_time
    global last_time_heuristic_accepted
    global unused_bisection_executions,stop_heuristic

    time_best_acc=0

    # For the bisection method: sample size, frequency and interpolation expression.
    df_sample_freq=pd.read_csv('results/data/general/sample_size_freq.csv',index_col=0)
    df_interpolation=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/df_Bisection.csv')
    sample_size=int(df_sample_freq[df_sample_freq['env_name']=='Turbines']['sample_size'])
    heuristic_freq=float(df_sample_freq[df_sample_freq['env_name']=='Turbines']['frequency_time'])
    interpolation_acc=list(df_interpolation['accuracy'])
    interpolation_time=list(df_interpolation['cost_per_eval'])
    lower_time=min(interpolation_time)
    upper_time=max(interpolation_time)

    # HEURISTIC I: The accuracy is updated using a constant frequency.
    if heuristic=='I': 
        if gen==0:
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=param)            
        else:
            if (time_acc+time_proc)-last_time_heuristic_accepted>=heuristic_freq:
                acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=param)

    # HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the last 
    # population is significantly different from the previous ones.
    if heuristic=='II': 
        if gen==0: 
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
            unused_bisection_executions=0
        else:

            # If the lasts optimal accuracies are higher than 0.9, the maximum accuracy will be considered as optimal from now on.
            if len(list_accuracies)>=param[1]+1:    
                if stop_heuristic==False:
                    prev_acc=list_accuracies[(-1-param[1]):-1]
                    prev_acc_high=np.array(prev_acc)>0.9
                    if sum(prev_acc_high)==param[1]:
                        stop_heuristic=True
                        acc=1
            
            if len(list_variances)>=param[0]+1 and stop_heuristic==False:
                # Calcular el intervalo de confianza.
                variance_q05=np.mean(list_variances[(-2-param[0]):-2])-2*np.std(list_variances[(-2-param[0]):-2])
                variance_q95=np.mean(list_variances[(-2-param[0]):-2])+2*np.std(list_variances[(-2-param[0]):-2])
                last_variance=list_variances[-1]
                
                # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
                if last_variance<variance_q05 or last_variance>variance_q95:

                    if (time_proc+time_acc)-last_time_heuristic_accepted>=heuristic_freq:   
                        unused_bisection_executions+=int((time_proc+time_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                        acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
                    else:
                        if unused_bisection_executions>0:
                            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
                            unused_bisection_executions-=1
   
    return acc,time_best_acc
    
#--------------------------------------------------------------------------------------------------
# Functions for the search process of the optimal turbine design.
#--------------------------------------------------------------------------------------------------
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

def learn(seed,heuristic,heuristic_param,accuracy=1,popsize=20):
    '''Run the CMA-ES algorithm with specific seed and using heuristic with the parameter value heuristic_param.'''

    global time_best_acc, last_time_heuristic_accepted,stop_heuristic
    
    # Initialize CMA-ES.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(6), 0.33,inopts={'bounds': [0, 1],'seed':seed,'popsize':popsize})

    # Initialize time counters.
    global time_proc,time_acc
    time_proc=0
    time_acc=0

    # Evaluate population designs until the maximum time is exhausted.
    stop_heuristic=False
    n_gen=0
    while time_proc+time_acc<max_time:

        # New population.
        solutions = es.ask()

        # Transform the scaled values of the parameters to the real values.
        list_turb_params=[transform_turb_params(x) for x in solutions]

        # Apply the heuristic.
        if n_gen==0:
            accuracy,time_best_acc=execute_heuristic(n_gen,accuracy,list_turb_params,seed,[],[],heuristic,heuristic_param)
        else:
            df_seed=pd.DataFrame(df)
            df_seed=df_seed[df_seed[1]==seed]
            accuracy,time_best_acc=execute_heuristic(n_gen,accuracy,list_turb_params,seed,list(df_seed[5]),list(df_seed[6]),heuristic,heuristic_param)


        # Obtain scores per evaluation and update time counters.
        list_scores=generation_score_list(list_turb_params,accuracy,count_time_gen=True)
        if time_best_acc!=0:
            time_acc-=time_best_acc
            last_time_heuristic_accepted=time_proc+time_acc
            readjustement=True
        else:
            readjustement=False

        # To build the following population.
        es.tell(solutions, list_scores)

        # Accumulate data of interest.
        test_score= fitness_function(transform_turb_params(es.result.xbest))
        df.append([heuristic_param,seed,n_gen,-test_score,readjustement,accuracy,np.var(list_scores),time_proc,time_acc,time_acc+time_proc])

        n_gen+=1
        
#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Grids.
list_seeds=range(2,102,1) # List of seeds.
list_h=[['II',[5,3]]]#[['II',[10,3]],['I',0.8],['I',0.95]] # List of heuristics (with their corresponding parameters).

# Parameters.
default_N=100
max_time=60*60 # 1h per seed.
heuristic_param=5

# Save database with information of interest associated with the training.
for h in list_h:
    heuristic=h[0]
    heuristic_param=h[1]

    df=[]
    for seed in tqdm(list_seeds):
        learn(seed,heuristic,heuristic_param)

    df=pd.DataFrame(df,columns=['heuristic_param','seed','n_gen','score','readjustement','accuracy','variance','elapsed_time_proc','elapsed_time_acc','elapsed_time'])
    df.to_csv('results/data/Turbines/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(heuristic_param)+'.csv')

# Join databases.
def join_df(heuristic,list_param):
    df=pd.read_csv('results/data/Turbines/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(param)+'.csv', index_col=0)
    for param in list_param[1:]:
        df_new=pd.read_csv('results/data/Turbines/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(param)+'.csv', index_col=0)
        df=pd.concat([df,df_new],ignore_index=True)

    df.to_csv('results/data/Turbines/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'.csv')

# join_df('I',[0.8,0.95])
# join_df('II',['[5, 3]','[10, 3]'])