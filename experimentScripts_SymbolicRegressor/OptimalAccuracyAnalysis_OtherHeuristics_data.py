'''
In this script, a total of 14 heuristics are implemented to automate the accuracy adjustment during
the training process. These heuristics are testing heuristics, which are used to find the appropriate
design to achieve the objective. The analysis done will be used to observe the advantages and disadvantages 
of all of them and finally to be able to design the appropriate heuristic with the highlighted features of 
the previous ones (this design is carried out in "OptimalAccuracyAnalysis.py"). The data obtained during the 
execution of each of the heuristics are stored and saved.
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
# For this code.
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pandas as pd
from tqdm import tqdm
import scipy as sc
import random

# For modifications made to borrowed code..
import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from gplearn._program import _Program
from gplearn.fitness import _fitness_map, _Fitness
from gplearn.functions import _function_map, _Function, sig1 as sigmoid
from gplearn.utils import _partition_estimators
from gplearn.utils import check_random_state

from gplearn.genetic import _parallel_evolve, MAX_INT
from gplearn.genetic import BaseSymbolic
import multiprocessing as mp
from itertools import combinations
import os

#==================================================================================================
# NEW FUNCTIONS
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Functions for the search process of the surface.
#--------------------------------------------------------------------------------------------------
# FUNCION 1
# Parametros:
#   >z_test: terceras coordenadas reales de los puntos de la superficie.
#   >z_pred: terceras coordenadas obtenidas a partir de la superficie predicha.
# Devuelve: el error absoluto medio de las dos listas anteriores.

def mean_abs_err(z_test,z_pred):
    '''Calculate the mean absolute error (MAE) between two vectors.'''
    mae=sum(abs(z_test-z_pred))/len(z_test)
    return mae

def build_pts_sample(n_sample,seed,expr_surf):
    '''
    Obtain a set of points belonging to a surface.

    Parameters
    ==========
    n_sample: Number of points to be constructed.
    seed: Seed for the random selection of points.
    expr_surf: Expression of the surface from which you want to extract the sample of points.

    Return
    ======
    Database with the three coordinates of the sample points.
    '''

    # Set seed.
    rng = check_random_state(seed)

    # Random grid for (x,y) coordinates.
    xy_sample=rng.uniform(-1, 1, n_sample*2).reshape(n_sample, 2)
    x=xy_sample[:,0]
    y=xy_sample[:,1]

    # Calculate corresponding heights (z values).
    z_sample=eval(expr_surf)

    # All data in a single array.
    pts_sample=np.insert(xy_sample, xy_sample.shape[1], z_sample, 1)

    return pts_sample

def select_pts_sample(pts_set,n_sample):
    '''Extract a sample from a set of points in an orderly manner.'''
    pts_sample=pts_set[:n_sample]
    return np.array(pts_sample)

def evaluate(df_test_pts,est_surf):
    '''
    Validating a surface (symbolic expression).

    Parameters
    ==========
    df_test_pts: Database with the three coordinates of the points that form the validation set. 
    est_surf: Surface selected in the GP execution process.

    Return
    ======
    The mean absolute error.
    '''

    # Split database with the coordinates of the points.
    xy_test=df_test_pts[:,[0,1]]
    z_test=df_test_pts[:,2]

    # Calculate the value of the third coordinates with the selected surface.
    z_pred=est_surf.predict(xy_test)

    # Calculate MAE associated to the set of points for the selected surface.
    score=mean_abs_err(z_test, z_pred)

    return score   

def learn(init_acc,train_seed,df_test_pts,heuristic,heuristic_param):
    '''
    Run the GP using a training seed and applying the indicated heuristic to adjust the accuracy during the process.

    Parameters
    ==========
    init_acc: Initial accuracy value.
    train_seed: Training seed.
    df_test_pts: Database with the three coordinates of the points that form the validation set.
    heuristic: Identifier of the heuristic that is being applied.
    heuristic_param: Parameter of the heuristic being considered.

    Return
    ======
    Selected surface.
    '''

    # Change predefined cardinal.
    train_n_pts=int(default_train_n_pts*init_acc)

    # Initialize training set.
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)

    # Definition of the GP with which the surface will be found.
    est_surf=SymbolicRegressor(random_state=train_seed)

    # Adjust the surface to the points.
    xy_train=df_train_pts[:,[0,1]]
    z_train=df_train_pts[:,2]
    est_surf.fit(init_acc,xy_train, z_train,train_seed,df_test_pts,heuristic,heuristic_param)    

    return est_surf._program 

#--------------------------------------------------------------------------------------------------
# Auxiliary functions to define the appropriate accuracy during execution process.
#--------------------------------------------------------------------------------------------------

def spearman_corr(x,y):
    '''Calculation of Spearman's correlation between two sequences.'''
    return sc.stats.spearmanr(x,y)[0]

def inverse_normalized_tau_kendall(x,y):
    '''Calculation of the normalized inverse tau kendall distance between two rankings.'''
    # Number of pairs with reverse order.
    pairs_reverse_order=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            case1 = x[i] < x[j] and y[i] > y[j]
            case2 = x[i] > x[j] and y[i] < y[j]

            if case1 or case2:
                pairs_reverse_order+=1  
    
    # Number of total pairs.
    total_pairs=len(list(combinations(x,2)))

    # Normalized tau kendall distance.
    tau_kendall=pairs_reverse_order/total_pairs

    return 1-tau_kendall

def idx_remove(list,idx):
    '''Remove from a list in one position element.'''
    new_list=[]
    for i in range(len(list)):
        if i!=idx:
            new_list.append(list[i])
    return new_list

def from_scores_to_ranking(list_scores):
    '''Convert score list to ranking list.'''
    list_pos_ranking=np.argsort(np.array(list_scores))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

def generation_score_list(list_surfaces,df_pts,
                          count_evaluations_acc=True,all_gen_evaluation=False,
                          gen_variance=False):
    '''
    Generate list with the scores associated to each surface that forms the generation.

    Parameters
    ==========
    list_surfaces: List with the expressions of the surfaces that form the generation.
    df_pts: Set of points on which each surface is to be evaluated.
    count_evaluations_acc: True or False in case you want to add or not the number of evaluations spent as extra evaluations, respectively. 
    all_gen_evaluation: True or False in case you want to add or not the number of evaluations spent as procedure evaluations, respectively. 
    gen_variance: True or False in case you want to return or not the variance of the generation scores.

    Return
    ======
    List of scores together with their variance (if specified).
    '''
    
    # Initialize score list.
    list_scores=[]

    # Split database with the coordinates of the points.
    X=df_pts[:,[0,1]]
    y=df_pts[:,2]

    # Evaluate each surface that forms the generation with the indicated accuracy.
    for expr_surf in list_surfaces:

        # Calculate the value of the third coordinates with the selected surface.
        y_pred=expr_surf.execute(X)

        # Calculate score associated to the set of points for the selected surface.
        score=mean_abs_err(y, y_pred)

        # Add score to the list.
        list_scores.append(score)

        # Update counters of given evaluations.
        if all_gen_evaluation:
            global n_evaluations
            n_evaluations+=len(y)
        else:
            if count_evaluations_acc:
                global n_evaluations_acc
                n_evaluations_acc+=len(y)
    
    if gen_variance:
        variance=np.var(list_scores)
        return list_scores,variance
    else:
        return list_scores

#--------------------------------------------------------------------------------------------------
# Functions associated with the bisection method for test heuristics 
#--------------------------------------------------------------------------------------------------
def bisection_method(init_acc,current_acc,list_surf_gen,train_seed,threshold,metric,
                     random_sample=True,fitness=None,first_gen=False,change_threshold='None'):
    
    '''
    Adapted implementation of the bisection method.

    Parameters
    ==========
    init_acc: Initial accuracy (the minimum considered).
    current_acc: Current accuracy (the most recently defined).
    list_surf_gen: List with the expressions of the surfaces that form the generation.
    train_seed: Training seed.
    threshold: Threshold of the bisection method with which the interval containing the optimal accuracy value will be updated.
    metric: 'spearman' or 'taukendall'.
    random_sample: True or False, in case you want to apply the bisection method with the 10% random or better of the surfaces that form the generation, respectively.
    fitness: None or list of scores associated with the generation.
    first_gen: True or False in case you are in the first or next iterations of the bisection method, respectively.
    change_threshold: 'None' in case you want to consider a constant value for the threshold; 'IncreasingMonotone' in case you want to update the threshold with 
    with the last value of the metric that has exceeded the previous threshold; 'NonMonotone' in case you want to update the threshold with the metric value associated with
    the accuracy selected as optimal.

    Return
    ======
    Accuracy value selected as optimal together with the new threshold for the bisection method (if specified).
    '''

    # Initialize lower and upper limit.
    acc0=init_acc
    acc1=1    

    # Midpoint.
    prev_m=current_acc
    m=(acc0+acc1)/2
    
    # Function to calculate the correlation between the rankings of the 10% random/best surfaces using the current and maximum accuracy.
    def similarity_between_current_best_acc(current_acc,acc,list_surf_gen,train_seed,metric,first_iteration,actual_n_evaluations_acc,random_sample,fitness):

        if random_sample:
            # Randomly select 10% of the surfaces that make up the generation.
            random.seed(train_seed)
            ind_surf=random.sample(range(len(list_surf_gen)),int(len(list_surf_gen)*0.1))
            list_surfaces=list(np.array(list_surf_gen)[ind_surf])


        else:
            # Select the best 10% of the surfaces forming the generation according to the accuracy of the previous generation.
            all_current_ranking=from_scores_to_ranking(fitness)
            if first_iteration:
                global n_evaluations_acc
                n_evaluations_acc+=int(default_train_n_pts*current_acc)*len(list_surf_gen)

            list_surfaces=list_surf_gen
            for ranking_pos in range(int(len(list_surf_gen)*0.1),len(list_surf_gen)):
                # Eliminate positions and surfaces that will not be used.
                ind_remove=all_current_ranking.index(ranking_pos)
                all_current_ranking.remove(ranking_pos)
                list_surfaces=idx_remove(list_surfaces,ind_remove)

        # Save the scores associated with each selected surface.
        best_scores=generation_score_list(list_surfaces,default_df_train_pts,count_evaluations_acc=first_iteration)# Con el maximo accuracy. 
        new_df_train_pts=select_pts_sample(default_df_train_pts,int(default_train_n_pts*acc))
        new_scores=generation_score_list(list_surfaces,new_df_train_pts)# Accuracy nuevo. 

        # Obtain vectors of associated rankings.
        new_ranking=from_scores_to_ranking(new_scores)# new accuracy. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximum accuracy. 
                
        # Compare two rankings.
        if metric=='spearman':
            metric_value=spearman_corr(new_ranking,best_ranking)
        if metric=='taukendall':
            metric_value=inverse_normalized_tau_kendall(new_ranking,best_ranking)

        return metric_value, n_evaluations_acc-actual_n_evaluations_acc

    # Readjust interval limits until the interval has a sufficiently small range or until the maximum number of evaluations is reached.
    global n_evaluations_acc
    first_iteration=True
    continue_bisection_method=True
    max_n_evaluations=default_train_n_pts*len(list_surf_gen) # Not to exceed the evaluations performed by default (accuracy=1)
    next_upper_threshold=[]

    while acc1-acc0>0.1 and continue_bisection_method:
        metric_value,extra_n_evaluations_acc=similarity_between_current_best_acc(current_acc,m,list_surf_gen,train_seed,metric,first_iteration,n_evaluations_acc,random_sample,fitness)
        if metric_value>=threshold:
            acc1=m

            next_upper_threshold.append(metric_value)

        else:
            acc0=m

        if first_gen and n_evaluations_acc+int(default_train_n_pts*m)*len(list_surf_gen)>max_n_evaluations:
            continue_bisection_method=False
            n_evaluations_acc-=extra_n_evaluations_acc
        else:
            prev_m=m
            m=(acc0+acc1)/2
        
        first_iteration=False
    
    if change_threshold=='IncreasingMonotone':
        if len(next_upper_threshold)==0:
            next_threshold=threshold
        else:
            next_threshold=next_upper_threshold[-1]
        return prev_m,next_threshold
    if change_threshold=='NonMonotone':
        next_threshold=metric_value
        return prev_m,next_threshold
    if change_threshold=='None':
        return prev_m

def set_initial_accuracy(init_acc,list_surf_gen,train_seed,metric,threshold=0.95,change_threshold='None',sample_size=None):
    '''
    Adjust with the bisection method the accuracy in the first generation.

    Parameters
    ==========
    init_acc: Initial accuracy (the minimum considered).
    list_surf_gen: List with the expressions of the surfaces forming the generation.
    train_seed: Training seed.
    metric: 'spearman' or 'taukendall'.
    threshold: Threshold of the bisection method with which the interval containing the optimal accuracy value will be updated.

    Returns 
    =======
    acc: Accuracy selected as optimal.
    X,y: Coordinates of the set of training points associated to the accuracy acc.
    fitness: List of scores of the generation calculated from the set of training points just set.
    acc_split: 10% of the remaining accuracy (1-acc).
    threshold: Threshold of the bisection method used.
    variance: Variance of the generation scores.
    '''

    # Calculate the minimum accuracy with which the maximum quality is obtained.
    if change_threshold !='None':
        acc,next_threshold=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,metric,first_gen=True,change_threshold=change_threshold,sample_size=sample_size)
    else:
        acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,metric,first_gen=True)

    # Calculate corresponding training set.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # Calculate the fitness vector of the generation using the defined accuracy.
    fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
    global n_evaluations_acc
    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

    # Define the accuracy increment to be used in the following iterations in case the accuracy needs to be increased.
    global acc_split
    acc_split=(1-acc)*0.1

    if change_threshold !='None':
        return acc,X,y,fitness,acc_split,threshold,next_threshold,variance
    else:
        return acc,X,y,fitness,acc_split,threshold,variance

#--------------------------------------------------------------------------------------------------
# Functions that implement different heuristics to update the accuracy.
#--------------------------------------------------------------------------------------------------

def update_accuracy_heuristic1(acc,X,y,population,fitness,train_seed,param):
    '''
    HEURISTIC 1 (Ascending accuracy)

    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Accuracy ajustment assessment (depending on parameter): It will be checked whether the correlation between rankings (current and maximum accuracy) 
    of random 10% of the surfaces of the generation exceeds or does not exceed a certain threshold (parameter). 
    >> Definition of new accuracy: Function dependent on the previous correlation.
    '''

    global n_evaluations_acc
    global n_evaluations

    if acc<1:
        # Function for accuracy increment calculation.
        def acc_split(corr,acc_rest,param):
            if param=='logistic':
                split=(1/(1+np.exp(12*(corr-0.5))))*acc_rest
            else:
                if corr<=param[0]:
                    split=acc_rest
                else:
                    split=-acc_rest*(((corr-param[0])/(1-param[0]))**(1/param[1]))+acc_rest
            return split
    
        # Randomly select 10% of the surfaces that make up the generation.
        random.seed(train_seed)
        ind_surf=random.sample(range(len(population)),int(len(population)*0.1))
        list_surfaces=list(np.array(population)[ind_surf])

        # Save the scores associated with each selected surface.
        best_scores=generation_score_list(list_surfaces,default_df_train_pts)# With maximum accuracy. 
        current_scores=list(np.array(fitness)[ind_surf])# Current accuracy.

        # Update the number of extra evaluations used to define the accuracy.
        n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)

        # Obtain associated rankings.
        current_ranking=from_scores_to_ranking(current_scores)# Current accuracy. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximum accuracy. 
                
        # Compare two rankings (calculate Spearman's correlation coefficient).
        corr=spearman_corr(current_ranking,best_ranking)

        # Depending on the similarity between the rankings calculate the split in accuracy for the next generation.
        split=acc_split(corr,1-acc,param)

        # Modify accuracy.
        prev_acc=acc
        acc=acc+split

        # Calculate new training set.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # If accuracy ascends and if it does not.
        if prev_acc!=acc:
            # Calculate the fitness vector of the generation using the defined accuracy.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            # Update number of process evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
            n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)
    else:
        # Update number of process evaluations.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

def update_accuracy_heuristic2(acc,init_acc,X,y,population,fitness,train_seed,param):
    '''
    HEURISTIC 2 (Ascending accuracy)
    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Accuracy adjustment assessment (depending on parameter): It will be checked if the correlation between rankings (current accuracy 
    and next in a predefined list) of random 10% of the surfaces of the generation exceeds or does not exceed a certain threshold (parameter). 
    >> Definition of new accuracy: current accuracy plus the constant split defined after applying the bisecting method (with threshold 0.95) at the beginning.
    '''
    global n_evaluations_acc
    global n_evaluations

    if acc<1:
        # Define a list of accuracies by doubling the value of the current accuracy successively up to the maximum.
        list_acc=[init_acc]
        next_acc=list_acc[-1]*2
        while next_acc<1:
            list_acc.append(next_acc)
            next_acc=list_acc[-1]*2
        if 1 not in list_acc:
            list_acc.append(1)
    
        # Randomly select 10% of the surfaces that make up the generation.
        random.seed(train_seed)
        ind_surf=random.sample(range(len(population)),int(len(population)*0.1))
        list_surfaces=list(np.array(population)[ind_surf])

        # Save the scores associated with each selected surface and calculate the ranking with the current accuracy.
        current_scores=list(np.array(fitness)[ind_surf])
        current_ranking=from_scores_to_ranking(current_scores)

        # Update the number of extra evaluations used to define the accuracy.
        n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)

        # As long as the correlation of the current ranking with the ranking associated with a higher accuracy is not lower 
        # than the threshold, continue testing with the rest of accuracies.
        possible_acc=list(np.array(list_acc)[np.array(list_acc)>acc])
        ind_next_acc=0
        corr=1
        while corr>param and ind_next_acc<len(possible_acc):

            # New set of points to evaluate surfaces.
            next_train_n_pts=int(default_train_n_pts*possible_acc[ind_next_acc])
            next_df_train_pts=select_pts_sample(default_df_train_pts,next_train_n_pts)

            # Save scores of the selected surfaces calculated with the following accuracy and obtain the corresponding ranking.
            next_scores=generation_score_list(list_surfaces,next_df_train_pts)
            next_ranking=from_scores_to_ranking(next_scores)

            # Compare two rankings (calculate Spearman's correlation coefficient).
            corr=spearman_corr(current_ranking,next_ranking)

            # Accuracy index update.
            ind_next_acc+=1
        
        # Modify accuracy.
        prev_acc=acc
        if corr<param:
            acc+=acc_split
            if acc>1:
                    acc=1

        # Calculate new training set.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # If accuracy ascends and if it does not.
        if prev_acc!=acc:
            # Calculate the fitness vector of the generation using the defined accuracy.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            # Update number of process evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
            n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)
    else:
        # Update number of process evaluations.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

def update_accuracy_heuristic3(acc,init_acc,X,y,population,fitness,param):
    '''
    HEURISTIC 3  (Ascending accuracy)
    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Accuracy adjustment assessment (depending on parameter): It will be checked whether the correlation between rankings
    (minimum and current accuracy) of the surfaces of the generation exceeds or does not exceed a certain threshold (parameter).
    >> Definition of new accuracy: Current accuracy plus the constant split defined after applying the bisecting method (with threshold 0.95) at the beginning.
    '''

    global n_evaluations_acc
    global n_evaluations

    if acc<1:
        # Save the scores associated with each surface.
        list_surfaces=list(population)
        worst_df_train_pts=select_pts_sample(default_df_train_pts,int(default_train_n_pts*init_acc))
        worst_scores=generation_score_list(list_surfaces,worst_df_train_pts)# Minimum accuracy. 
        current_scores=fitness# Current accuracy.

        # Update the number of extra evaluations used to define the accuracy.
        n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)
        
        # Obtain rankings.
        current_ranking=from_scores_to_ranking(current_scores)# Accuracy actual. 
        worst_ranking=from_scores_to_ranking(worst_scores)# Minimo accuracy. 
                
        # Compare two rankings (calculate Spearman's correlation coefficient).
        corr=spearman_corr(current_ranking,worst_ranking)

        # Depending on the similarity between the rankings consider a higher accuracy for the next generation.
        prev_acc=acc
        if corr>param:
            acc+=acc_split
            if acc>1:
                    acc=1

        # Calculate new training set.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # If accuracy ascends and if it does not.
        if prev_acc!=acc:
            # Calculate fitness list of the generation using the defined accuracy.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surfaces)
            n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)
    else:
        # Update number of process evaluations.
        n_evaluations+=default_train_n_pts*len(list(population))
    return acc,X,y,fitness

def update_accuracy_heuristic4(acc,X,y,population,fitness,param):
    '''
    HEURISTIC 4  (Ascending accuracy)
    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Accuracy adjustment assessment (depending on parameter): Every certain frequency (parameter) it is checked if the
    correlation between the rankings (current and maximum accuracy) of the best 10% of surfaces of the generation are equal or not.
    >> Definition of new accuracy: Current accuracy plus the constant split defined after applying the bisecting method (with threshold 0.95) at the beginning.
    '''

    global last_optimal_evaluations
    global n_evaluations_acc
    global n_evaluations
    if acc<1:
        if (n_evaluations+n_evaluations_acc)-last_optimal_evaluations>=param:

            # Assess whether the accuracy should be upgraded.
            list_surfaces=list(population)
            all_current_ranking=from_scores_to_ranking(fitness)

            n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)

            for ranking_pos in range(int(len(population)*0.1),len(population)):
                # Eliminate unused positions and surfaces.
                ind_remove=all_current_ranking.index(ranking_pos)
                all_current_ranking.remove(ranking_pos)
                list_surfaces=idx_remove(list_surfaces,ind_remove)
            current_ranking=all_current_ranking

            best_scores=generation_score_list(list_surfaces,default_df_train_pts) 
            best_ranking=from_scores_to_ranking(best_scores)

            corr=spearman_corr(current_ranking,best_ranking)

            # Define accuracy promotion in case it is to be promoted.
            prev_acc=acc
            if corr<1:
                acc+=acc_split
                if acc>1:
                    acc=1

            # Compute new training set.
            train_n_pts=int(default_train_n_pts*acc)
            df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
            X=df_train_pts[:,[0,1]]
            y=df_train_pts[:,2]
            
            # If accuracy is promoted and if it is not.
            if prev_acc!=acc:
                # Calculate fitness list of the generation using the defined accuracy.
                fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
            else:
                n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
                n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)

            # Update number of evaluations in which the accuracy has been adjusted.
            last_optimal_evaluations=n_evaluations+n_evaluations_acc
        else:
            # Update number of evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
    else:
        # Update number of process evaluations.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

def update_accuracy_heuristic5(acc,X,y,list_scores,population,fitness,param):
    '''
    HEURISTIC 5  (Ascending accuracy)
    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Accuracy adjustment assessment (depending on parameter): Look at the last score decrease in which position of the confidence
    interval it is located. The number of descents associated with the previous generations is the parameter to be defined. 
    These previous score decrements will be used to calculate the confidence interval.
    >> Definition of new accuracy: Current accuracy plus the constant split defined after applying the bisecting method (with threshold 0.95) at the beginning.
    '''

    global n_evaluations
    global n_evaluations_acc

    if acc<1:
        if len(list_scores)>param+1:

            # Function to calculate the confidence interval.
            def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                mean_list=[]
                for i in range(bootstrap_iterations):
                    sample = np.random.choice(data, len(data), replace=True) 
                    mean_list.append(np.mean(sample))
                return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

            # Calculate confidence interval of previous declines.
            list_scores1=list_scores[(-2-param):-2]
            list_scores2=list_scores[(-1-param):-1]

            list_score_falls=list(np.array(list_scores1)-np.array(list_scores2))
            conf_interval_q05,conf_interval_q95=bootstrap_confidence_interval(list_score_falls[0:-1])
            last_fall=list_score_falls[-1]

            # Update number of extra evaluations used for accuracy definition.
            n_evaluations_acc+=default_train_n_pts*(param+1)
            
            # Define increase of accuracy in case it is to be increased.
            prev_acc=acc
            if last_fall<conf_interval_q05:
                acc+=acc_split
                if acc>1:
                    acc=1

            # Calculate new training set.
            train_n_pts=int(default_train_n_pts*acc)
            df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
            X=df_train_pts[:,[0,1]]
            y=df_train_pts[:,2]

            # If accuracy ascends and if it does not.
            if prev_acc!=acc:
                # Calculate the fitness vector of the generation using the defined accuracy.
                fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
            else:
                # Update number of process evaluations.
                n_evaluations+=int(default_train_n_pts*acc)*len(list(population))

        else:
            # Update number of process evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
    else:
        # Update number of process evaluations.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

def update_accuracy_heuristic6(acc,list_surf_gen,train_seed,fitness,heuristic_param):
    '''
    HEURISTIC 6  (Optimal accuracy)
    >> Initial Accuracy: The one defined with the bisection method (the threshold is the parameter).
    >> Definition of new accuracy (depending on parameter): Apply the bisection method per generation
    with the random 10% of the solutions that form it (the threshold is the parameter) and using as lower 
    limit of the interval the accuracy of the previous iteration.
    '''
    # Calculate the minimum accuracy with which the maximum quality is obtained.
    prev_acc=acc
    acc=bisection_method(acc,acc,list_surf_gen,train_seed,heuristic_param,'spearman')

    # Calculate new training set.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # If accuracy ascends and if it does not.
    if prev_acc!=acc:
        # Calculate the fitness vector of the generation using the defined accuracy.
        fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
        global n_evaluations_acc
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
    else:
        # Update number of process evaluations.
        global n_evaluations
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)

    return acc,X,y,fitness

def update_accuracy_heuristic7(acc,init_acc,list_surf_gen,train_seed,fitness,heuristic_param):
    '''
    HEURISTIC 7 (Optimal accuracy)
    >> Initial Accuracy: The one defined with the bisection method (the threshold is the parameter).
    >> Definition of new accuracy (depending on parameter): Apply per generation the bisection method (the threshold 
    is the parameter) from zero (the lower limit of the interval is the minimum possible accuracy) with random 10% 
    of the surfaces forming the generation.
    '''
    global n_evaluations
    global n_evaluations_acc

    # Calculate the minimum accuracy with which the maximum quality is obtained.
    prev_acc=acc
    acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,heuristic_param,'spearman')

    # Calculate new training set.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # If the accuracy changes and if it does not.
    if prev_acc!=acc:
        # Calculate the fitness vector of the generation using the defined accuracy.
        fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        
    else:
        # Update number of process evaluations.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

    return acc,X,y,fitness

def update_accuracy_heuristic8(acc,init_acc,list_surf_gen,train_seed,fitness,heuristic_param):
    '''
    HEURISTIC 8 (Optimal accuracy)
    >> Initial Accuracy: The one defined with the bisection method (the threshold is the parameter).
    >> Definition of new accuracy (depending on parameter): Apply per generation the bisection method (the threshold 
    is the parameter) from zero with the best 10% of the surfaces that form the generation.
    '''
    global n_evaluations
    global n_evaluations_acc
    
    # Calculate the minimum accuracy with which the maximum quality is obtained.
    prev_acc=acc
    acc=bisection_method(init_acc,acc,list_surf_gen,train_seed,heuristic_param,'spearman',random_sample=False,fitness=fitness)

    # Calculate new training set.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # If the accuracy changes and if it does not.
    if prev_acc!=acc:
        # Calculate the fitness vector of the generation using the defined accuracy.
        fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
    else:
        # Update number of process evaluations.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

    return acc,X,y,fitness

def update_accuracy_heuristic9(acc,init_acc,X,y,list_surf_gen,train_seed,fitness,param):
    '''
    HEURISTIC 9 (Optimal accuracy)
    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Definition of new accuracy (depending on parameter): Every certain frequency (param) the bisection 
    method will be applied (with threshold 0.95) from zero with the random 10% of the surfaces that form the generation.
    '''

    global n_evaluations
    global n_evaluations_acc
    global last_optimal_evaluations

    if (n_evaluations+n_evaluations_acc)-last_optimal_evaluations>=param:

        # Calculate the minimum accuracy with which the maximum quality is obtained.
        prev_acc=acc
        acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,0.95,'spearman')

        # Calculate new training set.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # If the accuracy changes and if it does not.
        if prev_acc!=acc:
            # Calculate the fitness vector of the generation using the defined accuracy.
            fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            
        else:
            # Update number of process evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

        # Update number of evaluations in which the accuracy has been updated.
        last_optimal_evaluations=n_evaluations+n_evaluations_acc
    else:
        # Update number of evaluations.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)


    return acc,X,y,fitness

def update_accuracy_heuristic10(acc,init_acc,X,y,list_surf_gen,train_seed,fitness,param):
    '''
    HEURISTIC 10  (It is identical to heuristic 9 but uses the inverse normalized tau 
    Kendall distance instead of Spearman's correlation).
    '''
    global n_evaluations
    global n_evaluations_acc
    global last_optimal_evaluations
    if (n_evaluations+n_evaluations_acc)-last_optimal_evaluations>=param:

        # Calculate the minimum accuracy with which the maximum quality is obtained.
        prev_acc=acc
        acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,0.95,'taukendall')

        # Calculate new training set.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # If the accuracy changes and if it does not.
        if prev_acc!=acc:
            # Calculate the fitness vector of the generation using the defined accuracy.
            fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            
        else:
            # Update number of process evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

        # Update number of evaluations in which the accuracy has been updated.
        last_optimal_evaluations=n_evaluations+n_evaluations_acc
    else:
        # Update number of evaluations.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)


    return acc,X,y,fitness

def update_accuracy_heuristic11(init_acc,acc,X,y,list_scores,population,train_seed,fitness,param):
    '''
    HEURISTIC 11 (heuristic 7 with accuracy update frequency defined by heuristic 5)
    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Accuracy adjustment assessment (depending on parameter): Per generation it is checked if the new score decrease 
    is below the confidence interval of the previous param decreases.
    >> Definition of new accuracy: In case the accuracy has to be modified, the bisection method (threshold 0.95) 
    will be applied from zero with random 10% of the surfaces forming the generation.
    '''

    global n_evaluations

    if len(list_scores)>param+1:

        # Function to calculate the confidence interval.
        def bootstrap_confiance_interval(data,bootstrap_iterations=1000):
            mean_list=[]
            for i in range(bootstrap_iterations):
                sample = np.random.choice(data, len(data), replace=True) 
                mean_list.append(np.mean(sample))
            return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

        # Calculate confidence interval of previous declines.
        list_scores1=list_scores[(-2-param):-2]
        list_scores2=list_scores[(-1-param):-1]

        list_score_falls=list(np.array(list_scores1)-np.array(list_scores2))
        conf_interval_q05,conf_interval_q95=bootstrap_confiance_interval(list_score_falls[0:-1])
        last_fall=list_score_falls[-1]

        # Actualizar numero de evaluaciones extra empleadas para la definicion del accuracy.
        global n_evaluations_acc
        n_evaluations_acc+=default_train_n_pts*(param+1)
        
        # Definir ascenso de accuracy en caso de que se deba ascender.
        prev_acc=acc
        if last_fall<conf_interval_q05:
            acc=bisection_method(init_acc,init_acc,list(population),train_seed,0.95,'spearman')

        # Calculate new training set.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # If the accuracy changes and if it does not.
        if prev_acc!=acc:
            # Calculate the fitness vector of the generation using the defined accuracy.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            # Update number of process evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))

    else:
        # Update number of process evaluations.
        n_evaluations+=int(default_train_n_pts*acc)*len(list(population))


    return acc,X,y,fitness

def update_accuracy_heuristic12(acc,init_acc,X,y,list_variances,list_surf_gen,train_seed,fitness,param):
    '''
    HEURISTIC 12 (heuristic 7 with accuracy update frequency automatically defined)
    >> Initial Accuracy: The one defined with the bisection method (threshold 0.95).
    >> Accuracy adjustment assessment (depending on parameter): The variances of each generation calculated with 
    the optimal accuracy are stored, and with a certain amount of the most recent among them (param) a confidence 
    interval is calculated. If the last recorded variance is outside the interval, the accuracy will be readjusted.
    >> Definition of new accuracy: In case the accuracy has to be modified, the bisection method (threshold 0.95) 
    will be applied from zero with random 10% of the surfaces forming the generation.
    '''
        
    global n_evaluations
    global n_evaluations_acc
    threshold=None

    if len(list_variances)>=param+1:

        # Function to calculate the confidence interval.
        def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
            mean_list=[]
            for i in range(bootstrap_iterations):
                sample = np.random.choice(data, len(data), replace=True) 
                mean_list.append(np.mean(sample))
            return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

        variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-param):-2])
        last_variance=list_variances[-1]

        if last_variance<variance_q05 or last_variance>variance_q95:

            # Calculate the minimum accuracy with which the maximum quality is obtained.
            prev_acc=acc
            acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,0.95,'spearman')

            # Calculate new training set.
            train_n_pts=int(default_train_n_pts*acc)
            df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
            X=df_train_pts[:,[0,1]]
            y=df_train_pts[:,2]

            fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
        variance=np.var(fitness)

    return acc,X,y,fitness,threshold,variance

def update_accuraccy_heuristic13(acc,init_acc,X,y,list_variances,list_surf_gen,train_seed,fitness,threshold,param):
    '''
    HEURISTIC 13 (heuristic 12 with automatic monotonically ascending definition for the threshold of the bisection method) 
    Each time the accuracy (defined by heuristic 12) has to be readjusted, the threshold to be considered in the bisection 
    method for the next readjustment of the accuracy will be recalculated. The new threshold will be the Spearman correlation 
    at which the current threshold was last exceeded when applying the bisection method. In case it is not exceeded the threshold 
    will be maintained.
    '''
    global n_evaluations
    global n_evaluations_acc
    next_threshold=threshold

    if threshold<1:

        if len(list_variances)>=param+1:

            # Function to calculate the confidence interval.
            def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                mean_list=[]
                for i in range(bootstrap_iterations):
                    sample = np.random.choice(data, len(data), replace=True) 
                    mean_list.append(np.mean(sample))
                return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

            variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-param):-2])
            last_variance=list_variances[-1]

            if last_variance<variance_q05 or last_variance>variance_q95:

                # Calculate the minimum accuracy with which the maximum quality is obtained.
                prev_acc=acc
                acc,next_threshold=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,'spearman',change_threshold='IncreasingMonotone')

                # Calculate new training set.
                train_n_pts=int(default_train_n_pts*acc)
                df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                X=df_train_pts[:,[0,1]]
                y=df_train_pts[:,2]

                fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        
                # If the accuracy changes and if it does not.
                if prev_acc!=acc:
                    # Calculate the fitness vector of the generation using the defined accuracy.
                    fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
                    
                else:
                    # Update number of process evaluations.
                    n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            else:
                n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                variance=np.var(fitness)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        n_evaluations+=default_train_n_pts*len(list_surf_gen)
        variance=np.var(fitness)

    return acc,X,y,fitness,threshold,next_threshold,variance

def update_accuraccy_heuristic14(acc,init_acc,X,y,list_variances,list_surf_gen,train_seed,fitness,threshold,param):
    '''
    HEURISTIC 14 (heuristic 12 with automatic non-monotonic definition for the threshold of the bisection method)
    Each time the accuracy (defined by heuristic 12) has to be readjusted, the threshold to be considered in the 
    bisection method for the next readjustment of the accuracy will be recalculated. The new threshold will be the 
    Spearman correlation associated to the accuracy selected as optimal in the bisection method.
    '''
    global n_evaluations
    global n_evaluations_acc
    next_threshold=threshold

    if threshold<1:

        if len(list_variances)>=param+1:

            # Function to calculate the confidence interval.
            def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                mean_list=[]
                for i in range(bootstrap_iterations):
                    sample = np.random.choice(data, len(data), replace=True) 
                    mean_list.append(np.mean(sample))
                return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

            variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-param):-2])
            last_variance=list_variances[-1]

            if last_variance<variance_q05 or last_variance>variance_q95:

                # Calculate the minimum accuracy with which the maximum quality is obtained.
                prev_acc=acc
                acc,next_threshold=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,'spearman',change_threshold='NonMonotone')

                # Calculate new training set.
                train_n_pts=int(default_train_n_pts*acc)
                df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                X=df_train_pts[:,[0,1]]
                y=df_train_pts[:,2]

                fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        
                # If the accuracy changes and if it does not.
                if prev_acc!=acc:
                    # Calculate the fitness vector of the generation using the defined accuracy.
                    fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
                    
                else:
                    # Update number of process evaluations.
                    n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            else:
                n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                variance=np.var(fitness)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        n_evaluations+=default_train_n_pts*len(list_surf_gen)
        variance=np.var(fitness)

    return acc,X,y,fitness,threshold,next_threshold,variance

def execute_heuristic(heuristic,heuristic_param,train_seed,gen,population,init_acc,acc,X,y,fitness):
    '''Apply the indicated heuristic. This function is called from the function fit (modified by new_fit).'''
    global train_pts_seed
    global last_optimal_evaluations
    global acc_split
    global next_threshold
    global sample_size,last_time_heuristic_accepted
    global unused_bisection_executions
    global heuristic_accepted

    threshold=None
    variance=None

    # Set accuracy of the initial generation.
    if gen==0:
        heuristic_accepted=True
        if heuristic in [1,2,3,4,5,9,11,12]:
            if heuristic==11:
                train_pts_seed=gen
            acc,X,y,fitness,acc_split,threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman')
        if heuristic in [6,7,8]:
            acc,X,y,fitness,acc_split,threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman',threshold=heuristic_param)
        if heuristic==10:
            acc,X,y,fitness,acc_split,threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'taukendall')
        if heuristic==13:
            acc,X,y,fitness,acc_split,threshold,next_threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman',threshold=heuristic_param[0],change_threshold='IncreasingMonotone')
        if heuristic==14:
            acc,X,y,fitness,acc_split,threshold,next_threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman',threshold=heuristic_param[0],change_threshold='NonMonotone')

        # Update the last number of evaluations in which the accuracy has been updated for the heuristics that need it.   
        if heuristic in [4,9,10]:
            last_optimal_evaluations=n_evaluations+n_evaluations_acc
       
    # Update accuracy in the rest of the generations.
    else:
        if heuristic==1:
            acc,X,y,fitness=update_accuracy_heuristic1(acc,X,y,population,fitness,train_seed,heuristic_param)               
        if heuristic==2:
            acc,X,y,fitness=update_accuracy_heuristic2(acc,init_acc,X,y,population,fitness,train_seed,heuristic_param)
        if heuristic==3:
            acc,X,y,fitness=update_accuracy_heuristic3(acc,init_acc,X,y,population,fitness,heuristic_param)
        if heuristic==4:
            acc,X,y,fitness=update_accuracy_heuristic4(acc,X,y,population,fitness,heuristic_param)
        if heuristic==5:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness=update_accuracy_heuristic5(acc,X,y,list(df_seed[6]),population,fitness,heuristic_param)
        if heuristic==6:
            acc,X,y,fitness=update_accuracy_heuristic6(acc,list(population),train_seed,fitness,heuristic_param)
        if heuristic==7:
            acc,X,y,fitness=update_accuracy_heuristic7(acc,init_acc,list(population),train_seed,fitness,heuristic_param)
        if heuristic==8:
            acc,X,y,fitness=update_accuracy_heuristic8(acc,init_acc,list(population),train_seed,fitness,heuristic_param)
        if heuristic==9:
            acc,X,y,fitness=update_accuracy_heuristic9(acc,init_acc,X,y,list(population),train_seed,fitness,heuristic_param)
        if heuristic==10:
            acc,X,y,fitness=update_accuracy_heuristic10(acc,init_acc,X,y,list(population),train_seed,fitness,heuristic_param)
        if heuristic==11:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness=update_accuracy_heuristic11(init_acc,acc,X,y,list(df_seed[6]),population,train_seed,fitness,heuristic_param)
        if heuristic==12:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,variance=update_accuracy_heuristic12(acc,init_acc,X,y,list(df_seed[3]),list(population),train_seed,fitness,heuristic_param)
        if heuristic==13:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,next_threshold,variance=update_accuraccy_heuristic13(acc,init_acc,X,y,list(df_seed[3]),list(population),train_seed,fitness,next_threshold,heuristic_param[1])
        if heuristic==14:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,next_threshold,variance=update_accuraccy_heuristic14(acc,init_acc,X,y,list(df_seed[3]),list(population),train_seed,fitness,next_threshold,heuristic_param[1])
    
    return acc,X,y,fitness,threshold,variance

#==================================================================================================
# FUNCTIONS DESIGNED FROM SOME EXISTING ONES
#==================================================================================================
def new_raw_fitness(self, X, y, sample_weight):
    '''This function replaces the existing raw_fitness function.'''
    y_pred = self.execute(X)
    if self.transformer:
        y_pred = self.transformer(y_pred)
    raw_fitness = self.metric(y, y_pred, sample_weight)
    
    # MODIFICATION: Add the number of evaluations performed (as many as the number of points in the training set).
    if count_evaluations:
        global n_evaluations
        n_evaluations+=X.shape[0]

    return raw_fitness

def new_parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    '''This function replaces the existing _parallel_evolve function.'''
   
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']

    max_samples = int(max_samples * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []
    i=0# MODIFICATION: initialize counter manually.
    while i<n_programs and n_evaluations<max_n_eval:#MODIFICATION: add new constraint to end the loop.

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        
        
        program.raw_fitness_=program.raw_fitness(X, y, curr_sample_weight)
         
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_= program.raw_fitness(X, y, oob_sample_weight)
            

        programs.append(program)

        i+=1# MODIFICATION: update counter manually.
    return programs

def find_best_individual_final_generation(self,fitness):
    '''
    This function matches the original find_best_individual_final_generation function. It is explicitly
    added in this script because it is called in the following function.
    '''

    if isinstance(self, TransformerMixin):
        # Find the best individuals in the final generation
        fitness = np.array(fitness)
        if self._metric.greater_is_better:
            hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
        else:
            hall_of_fame = fitness.argsort()[:self.hall_of_fame]
        evaluation = np.array([gp.execute(X) for gp in
                                [self._programs[-1][i] for
                                i in hall_of_fame]])
        if self.metric == 'spearman':
            evaluation = np.apply_along_axis(rankdata, 1, evaluation)

        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.abs(np.corrcoef(evaluation))
        np.fill_diagonal(correlations, 0.)
        components = list(range(self.hall_of_fame))
        indices = list(range(self.hall_of_fame))
        # Iteratively remove least fit individual of most correlated pair
        while len(components) > self.n_components:
            most_correlated = np.unravel_index(np.argmax(correlations),
                                                correlations.shape)
            # The correlation matrix is sorted by fitness, so identifying
            # the least fit of the pair is simply getting the higher index
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))
        self._best_programs = [self._programs[-1][i] for i in
                                hall_of_fame[components]]

    else:
        # Find the best individual in the final generation
        if self._metric.greater_is_better:
            self._program = self._programs[-1][np.argmax(fitness)]
        else:
            self._program = self._programs[-1][np.argmin(fitness)]

def new_fit(self,init_acc, X, y, train_seed,df_test_pts,heuristic,heuristic_param,sample_weight=None):# MODIFICATION: add new arguments.
    '''This function replaces the existing fit function.'''

    random_state = check_random_state(self.random_state)

    # Check arrays
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    if isinstance(self, ClassifierMixin):
        X, y = self._validate_data(X, y, y_numeric=False)
        check_classification_targets(y)

        if self.class_weight:
            if sample_weight is None:
                sample_weight = 1.
            # modify the sample weights with the corresponding class weight
            sample_weight = (sample_weight *
                                compute_sample_weight(self.class_weight, y))

        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes != 2:
            raise ValueError("y contains %d class after sample_weight "
                                "trimmed classes with zero weights, while 2 "
                                "classes are required."
                                % n_trim_classes)
        self.n_classes_ = len(self.classes_)

    else:
        X, y = self._validate_data(X, y, y_numeric=True)

    hall_of_fame = self.hall_of_fame
    if hall_of_fame is None:
        hall_of_fame = self.population_size
    if hall_of_fame > self.population_size or hall_of_fame < 1:
        raise ValueError('hall_of_fame (%d) must be less than or equal to '
                            'population_size (%d).' % (self.hall_of_fame,
                                                    self.population_size))
    n_components = self.n_components
    if n_components is None:
        n_components = hall_of_fame
    if n_components > hall_of_fame or n_components < 1:
        raise ValueError('n_components (%d) must be less than or equal to '
                            'hall_of_fame (%d).' % (self.n_components,
                                                    self.hall_of_fame))

    self._function_set = []
    for function in self.function_set:
        if isinstance(function, str):
            if function not in _function_map:
                raise ValueError('invalid function name %s found in '
                                    '`function_set`.' % function)
            self._function_set.append(_function_map[function])
        elif isinstance(function, _Function):
            self._function_set.append(function)
        else:
            raise ValueError('invalid type %s found in `function_set`.'
                                % type(function))
    if not self._function_set:
        raise ValueError('No valid functions found in `function_set`.')

    # For point-mutation to find a compatible replacement node
    self._arities = {}
    for function in self._function_set:
        arity = function.arity
        self._arities[arity] = self._arities.get(arity, [])
        self._arities[arity].append(function)

    if isinstance(self.metric, _Fitness):
        self._metric = self.metric
    elif isinstance(self, RegressorMixin):
        if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                'spearman', 'spearman'):
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]
    elif isinstance(self, ClassifierMixin):
        if self.metric != 'log loss':
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]
    elif isinstance(self, TransformerMixin):
        if self.metric not in ('spearman', 'spearman'):
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]

    self._method_probs = np.array([self.p_crossover,
                                    self.p_subtree_mutation,
                                    self.p_hoist_mutation,
                                    self.p_point_mutation])
    self._method_probs = np.cumsum(self._method_probs)

    if self._method_probs[-1] > 1:
        raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                            'p_hoist_mutation and p_point_mutation should '
                            'total to 1.0 or less.')

    if self.init_method not in ('half and half', 'grow', 'full'):
        raise ValueError('Valid program initializations methods include '
                            '"grow", "full" and "half and half". Given %s.'
                            % self.init_method)

    if not((isinstance(self.const_range, tuple) and
            len(self.const_range) == 2) or self.const_range is None):
        raise ValueError('const_range should be a tuple with length two, '
                            'or None.')

    if (not isinstance(self.init_depth, tuple) or
            len(self.init_depth) != 2):
        raise ValueError('init_depth should be a tuple with length two.')
    if self.init_depth[0] > self.init_depth[1]:
        raise ValueError('init_depth should be in increasing numerical '
                            'order: (min_depth, max_depth).')

    if self.feature_names is not None:
        if self.n_features_in_ != len(self.feature_names):
            raise ValueError('The supplied `feature_names` has different '
                                'length to n_features. Expected %d, got %d.'
                                % (self.n_features_in_,
                                len(self.feature_names)))
        for feature_name in self.feature_names:
            if not isinstance(feature_name, str):
                raise ValueError('invalid type %s found in '
                                    '`feature_names`.' % type(feature_name))

    if self.transformer is not None:
        if isinstance(self.transformer, _Function):
            self._transformer = self.transformer
        elif self.transformer == 'sigmoid':
            self._transformer = sigmoid
        else:
            raise ValueError('Invalid `transformer`. Expected either '
                                '"sigmoid" or _Function object, got %s' %
                                type(self.transformer))
        if self._transformer.arity != 1:
            raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                'got %d.' % (self._transformer.arity))

    params = self.get_params()
    params['_metric'] = self._metric
    if hasattr(self, '_transformer'):
        params['_transformer'] = self._transformer
    else:
        params['_transformer'] = None
    params['function_set'] = self._function_set
    params['arities'] = self._arities
    params['method_probs'] = self._method_probs

    if not self.warm_start or not hasattr(self, '_programs'):
        # Free allocated memory, if any
        self._programs = []
        self.run_details_ = {'generation': [],
                                'average_length': [],
                                'average_fitness': [],
                                'best_length': [],
                                'best_fitness': [],
                                'best_oob_fitness': [],
                                'generation_time': []}

    prior_generations = len(self._programs)
    n_more_generations = self.generations - prior_generations

    if n_more_generations < 0:
        raise ValueError('generations=%d must be larger or equal to '
                            'len(_programs)=%d when warm_start==True'
                            % (self.generations, len(self._programs)))
    elif n_more_generations == 0:
        fitness = [program.raw_fitness_ for program in self._programs[-1]]
        warn('Warm-start fitting without increasing n_estimators does not '
                'fit new programs.')

    if self.warm_start:
        # Generate and discard seeds that would have been produced on the
        # initial fit call.
        for i in range(len(self._programs)):
            _ = random_state.randint(MAX_INT, size=self.population_size)

    if self.verbose:
        # Print header fields
        self._verbose_reporter()

    start_total_time=time() #MODIFICATION: start counting training time.
    gen=0# MODIFICATION: so that the procedure does not end when a number of generations is reached, the generations are counted with an independent counter.

    # MODIFICATION: global variable that will count the number of evaluations carried out, understanding by evaluation each evaluation of a point in an expression of a surface.
    global n_evaluations
    n_evaluations=0
    global n_evaluations_acc
    n_evaluations_acc=0
    acc=init_acc
    global count_evaluations
    count_evaluations=False# So that in the first generation the evaluations made when calculating the population are not counted.
    
    global train_pts_seed
    while n_evaluations+n_evaluations_acc < max_n_eval:# MODIFICATION: modify the training limit.
        
        start_time = time()

        if gen == 0:
            parents = None
        else:
            parents = self._programs[gen - 1]

        # Parallel loop
        n_jobs, n_programs, starts = _partition_estimators(
            self.population_size, self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.population_size)

        population = Parallel(n_jobs=n_jobs,
                                verbose=int(self.verbose > 1))(
            delayed(_parallel_evolve)(n_programs[i],
                                        parents,
                                        X,
                                        y,
                                        sample_weight,
                                        seeds[starts[i]:starts[i + 1]],
                                        params)
            for i in range(n_jobs))

        # Reduce, maintaining order across different n_jobs
        population = list(itertools.chain.from_iterable(population))
        fitness = [program.raw_fitness_ for program in population]

        # MODIFICATION: apply heuristic.
        if heuristic !='None':
            acc,X,y,fitness,threshold,variance=execute_heuristic(heuristic,heuristic_param,train_seed,gen,population,init_acc,acc,X,y,fitness)

        length = [program.length_ for program in population]

        parsimony_coefficient = None
        if self.parsimony_coefficient == 'auto':
            parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                        np.var(length))
        for program in population:
            program.fitness_ = program.fitness(parsimony_coefficient)

        self._programs.append(population)

        # Remove old programs that didn't make it into the new population.
        if not self.low_memory:
            for old_gen in np.arange(gen, 0, -1):
                indices = []
                for program in self._programs[old_gen]:
                    if program is not None:
                        for idx in program.parents:
                            if 'idx' in idx:
                                indices.append(program.parents[idx])
                indices = set(indices)
                for idx in range(self.population_size):
                    if idx not in indices:
                        self._programs[old_gen - 1][idx] = None
        elif gen > 0:
            # Remove old generations
            self._programs[gen - 1] = None

        # Record run details
        if self._metric.greater_is_better:
            best_program = population[np.argmax(fitness)]
        else:
            best_program = population[np.argmin(fitness)]

        self.run_details_['generation'].append(gen)
        self.run_details_['average_length'].append(np.mean(length))
        self.run_details_['average_fitness'].append(np.mean(fitness))
        self.run_details_['best_length'].append(best_program.length_)
        self.run_details_['best_fitness'].append(best_program.raw_fitness_)
        oob_fitness = np.nan
        if self.max_samples < 1.0:
            oob_fitness = best_program.oob_fitness_
        self.run_details_['best_oob_fitness'].append(oob_fitness)
        generation_time = time() - start_time
        self.run_details_['generation_time'].append(generation_time)

        if self.verbose:
            self._verbose_reporter(self.run_details_)

        # Check for early stopping
        if self._metric.greater_is_better:
            best_fitness = fitness[np.argmax(fitness)]
        else:
            best_fitness = fitness[np.argmin(fitness)]

   
        find_best_individual_final_generation(self,fitness) # MODIFICATION: to be able to evaluate the best surface during the process.
        
        # MODIFICATION: save the data of interest during the training.
        global heuristic_accepted 
        score=evaluate(df_test_pts,self)
        elapsed_time=time()-start_total_time   
        df_train.append([heuristic_param,train_seed,threshold,variance,acc,gen,score,elapsed_time,generation_time,n_evaluations,n_evaluations_acc,n_evaluations+n_evaluations_acc])

        gen+=1# MODIFICATION: update number of generations.

    find_best_individual_final_generation(self,fitness)# MODIFICATION: to obtain the best individual of the last generation.
    
    return self

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# To use the modified fit function.
_Program.raw_fitness=new_raw_fitness
_parallel_evolve=new_parallel_evolve
BaseSymbolic.fit=new_fit

# Original surface.
expr_surf_real='x**2-y**2+y-1'

# List of train seeds.
list_train_seeds=range(1,101,1)

# Training set and parameters.
default_train_n_pts=50# Cardinal of predefined initial set.
train_pts_seed=0
default_df_train_pts=build_pts_sample(default_train_n_pts,train_pts_seed,expr_surf_real)
max_n_eval=20*50*1000# Equivalent to 20 generations with maximum accuracy.

# Validation set and parameters.
test_n_pts=default_train_n_pts
test_pts_seed=1
df_test_pts=build_pts_sample(test_n_pts,test_pts_seed,expr_surf_real)

# Function to perform parallel execution.
def parallel_processing(arg):

    # Extract heuristic information from the function argument.
    heuristic=arg[0]
    param=arg[1]

    # To save training data.
    global df_train
    df_train=[]

    # Initial accuracy.
    init_acc=1/default_train_n_pts# The one corresponding to a set formed by a single point.

    # Fill in database.
    for train_seed in tqdm(list_train_seeds):
        learn(init_acc,train_seed,df_test_pts,heuristic,param)     

    # Save constructed database.
    df_train=pd.DataFrame(df_train,columns=['heuristic_param','train_seed','threshold','variance','acc','n_gen','score','elapsed_time','time_gen','n_eval_proc','n_eval_acc','n_eval'])
    df_train.to_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis_OtherHeuristics/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'_param'+str(param)+'.csv')
    

# Prepare function arguments for parallel processing.
# Argument form: [heuristic identifier, parameter value].
list_arg=[
    [1,(0,1)],[1,(0,3)],[1,(0.5,3)],[1,(0.5,1)],[1,(0,0.3)],[1,'logistic'],
    [2,0.8],[2,0.6],
    [3,0.8],[3,0.6],
    [4,100000],[4,500000],[4,50000],
    [5,10],[5,20],
    [6,0.95],[6,0.8],
    [7,0.95],[7,0.8],
    [8,0.95],[8,0.8],
    [9,100000],[9,500000],
    [10,100000],[10,500000],
    [11,5],[11,10],
    [12,5],[12,10],
    [13,(0.5,5)],[13,(0.5,10)],[13,(0.95,5)],[13,(0.95,10)],[13,(0.8,5)],[13,(0.8,10)],
    [14,(0.5,5)],[14,(0.5,10)],[14,(0.95,5)],[14,(0.95,10)],[14,(0.8,5)],[14,(0.8,10)]
    ]

# Parallel processing.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,list_arg)
pool.close()

# Grouping databases
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
                df=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis_Otherheuristics/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/SymbolicRegressor/OptimalAccuracyAnalysis_Otherheuristics/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv')
                first=False
            else:
                df_new=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis_Otherheuristics/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/SymbolicRegressor/OptimalAccuracyAnalysis_Otherheuristics/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv')
                df=pd.concat([df,df_new],ignore_index=True)


        df.to_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis_Otherheuristics/df_train_OptimalAccuracy_heuristic'+str(key)+'.csv')


concat_same_heuristic_df(list_arg)



