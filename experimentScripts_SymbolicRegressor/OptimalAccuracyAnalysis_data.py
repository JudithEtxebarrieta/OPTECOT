'''
In this script the two main heuristics proposed to automate the accuracy adjustment during the 
execution process are implemented. The data obtained during the execution of each of the 
heuristics are stored and saved.

The general descriptions of the heuristics are:
HEURISTIC I: The accuracy is updated using the constant frequency calculated in experimentScripts_general/sample_size_bisection_method.py.
HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the last population is significantly different from the previous ones.
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

# For modifications made to borrowed code.
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
# Functions for the learning process or surface search.
#--------------------------------------------------------------------------------------------------
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
    '''Obtain subsample formed by the first n_sample of a set of points.'''
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
# Functions associated with the bisection method
#--------------------------------------------------------------------------------------------------
def customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,threshold,sample_size,interpolation_pts):
    '''Adapted implementation of bisection method'''

    # Initialize lower and upper limit.
    time0=lower_time
    time1=upper_time    

    # Midpoint.
    prev_m=lower_time
    m=(time0+time1)/2
    
    # Function to calculate the correlation between the rankings of the sample_size surfaces using the current and maximum accuracy.
    def similarity_between_current_best_acc(time,list_surf_gen,train_seed,first_iteration):

        # Randomly select sample_size surfaces forming the generation.
        random.seed(train_seed)
        ind_surf=random.sample(range(len(list_surf_gen)),sample_size)
        list_surfaces=list(np.array(list_surf_gen)[ind_surf])

        # Save the scores associated with each selected surface.
        best_scores=generation_score_list(list_surfaces,default_df_train_pts,count_evaluations_acc=first_iteration)# With maximum accuracy. 
        new_df_train_pts=select_pts_sample(default_df_train_pts,int(default_train_n_pts*time))
        new_scores=generation_score_list(list_surfaces,new_df_train_pts)# New accuracy. 

        # Obtain associated rankings.
        new_ranking=from_scores_to_ranking(new_scores)# Accuracy nuevo. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximo accuracy. 
                
        # Compare two rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)

        return metric_value

    # Reset interval limits until the interval has a sufficiently small range (10% of the maximum length).
    global n_evaluations_acc
    first_iteration=True
    stop_threshold=(time1-time0)*0.1

    while time1-time0>stop_threshold:
        metric_value=similarity_between_current_best_acc(np.interp(m,interpolation_pts[0],interpolation_pts[1]),list_surf_gen,train_seed,first_iteration)
        if metric_value>=threshold:
            time1=m
        else:
            time0=m

        prev_m=m
        m=(time0+time1)/2
        
        first_iteration=False


    return np.interp(prev_m,interpolation_pts[0],interpolation_pts[1])

def customized_set_initial_accuracy(lower_time,upper_time,list_surf_gen,train_seed,sample_size,interpolation_pts,threshold=0.95):
    '''Adjusting with the bisection method the accuracy in the first generation.'''

    # Calculate the minimum accuracy with which the maximum quality is obtained.
    acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,threshold,sample_size,interpolation_pts)

    # Calculate corresponding training set.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # Calculate the fitness vector of the generation using the defined accuracy.
    fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
    global n_evaluations_acc
    n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

    return acc,X,y,fitness,threshold,variance

#--------------------------------------------------------------------------------------------------
# Functions that implement different heuristics to update the accuracy.
#--------------------------------------------------------------------------------------------------

def update_accuracy_heuristicI(acc,lower_time,upper_time,X,y,list_surf_gen,train_seed,fitness,heuristic_param,sample_size,heuristic_freq,interpolation_pts):
    '''
    HEURISTIC I: The accuracy is updated using the constant frequency calculated in 
    experimentScripts_general/sample_size_bisection_method.py.
    '''
    
    global n_evaluations
    global n_evaluations_acc
    global last_time_heuristic_accepted,heuristic_accepted
    heuristic_accepted=False

    if (n_evaluations+n_evaluations_acc)-last_time_heuristic_accepted>=heuristic_freq:
        heuristic_accepted=True
        # Calculate the minimum accuracy with which the maximum quality is obtained.
        prev_acc=acc
        acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,heuristic_param,sample_size,interpolation_pts)

        # Calculate new training set.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # If the accuracy changes and if it does not.
        if prev_acc!=acc:
            # Calculate the fitness vector of the generation using the defined accuracy.
            fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size
            
        else:
            # Update number of process evaluations.
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

        last_time_heuristic_accepted=n_evaluations+n_evaluations_acc
    else:
        # Update number of process evaluations.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)

    return acc,X,y,fitness



def update_accuracy_heuristicII(acc,lower_time,upper_time,X,y,list_accuracies,list_variances,list_surf_gen,train_seed,fitness,param,sample_size,heuristic_freq,interpolation_pts):
    '''
    HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the
    last population is significantly different from the previous ones. In addition, when it is observed 
    that in the last populations the optimum accuracy considered is higher than 0.9, the accuracy will 
    no longer be adjusted and the maximum accuracy will be considered for the following populations.
    '''
    
    global n_evaluations
    global n_evaluations_acc
    global last_time_heuristic_accepted,heuristic_accepted
    global unused_bisection_executions, stop_heuristic

    threshold=None
    heuristic_accepted=False

    # If the last optimal accuracy is higher than 0.9, the maximum accuracy will be considered as optimal from now on.
    if len(list_accuracies)>=param[1]:
        if stop_heuristic==True:
            n_evaluations+=int(default_train_n_pts)*len(list_surf_gen)
            variance=np.var(fitness)
    
        if stop_heuristic==False:
            prev_acc=list_accuracies[(-1-param[1]):-1]
            prev_acc_high=np.array(prev_acc)>0.9
            if sum(prev_acc_high)==param[1]:
                stop_heuristic=True

                acc=1
                X=default_df_train_pts[:,[0,1]]
                y=default_df_train_pts[:,2]
                fitness,variance=generation_score_list(list_surf_gen,default_df_train_pts,all_gen_evaluation=True,gen_variance=True) 

    if len(list_variances)>=param[0]+1 and stop_heuristic==False:

        # Calculate the confidence interval.
        variance_q05=np.mean(list_variances[(-2-param[0]):-2])-2*np.std(list_variances[(-2-param[0]):-2])
        variance_q95=np.mean(list_variances[(-2-param[0]):-2])+2*np.std(list_variances[(-2-param[0]):-2])
        last_variance=list_variances[-1]

        if last_variance<variance_q05 or last_variance>variance_q95:
            if (n_evaluations+n_evaluations_acc)-last_time_heuristic_accepted>=heuristic_freq:
                heuristic_accepted=True

                unused_bisection_executions+=int((n_evaluations+n_evaluations_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                # Calculate the minimum accuracy with which the maximum quality is obtained.
                acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,0.95,sample_size,interpolation_pts)

                # Calculate new training set.
                train_n_pts=int(default_train_n_pts*acc)
                df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                X=df_train_pts[:,[0,1]]
                y=df_train_pts[:,2]

                fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

                last_time_heuristic_accepted=n_evaluations+n_evaluations_acc
            else:
                if unused_bisection_executions>0:
                    heuristic_accepted=True
                    
                    # Calculate the minimum accuracy with which the maximum quality is obtained..
                    acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,0.95,sample_size,interpolation_pts)

                    # Calculate new training set.
                    train_n_pts=int(default_train_n_pts*acc)
                    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                    X=df_train_pts[:,[0,1]]
                    y=df_train_pts[:,2]

                    fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                    n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

                    unused_bisection_executions-=1
                    last_time_heuristic_accepted=n_evaluations+n_evaluations_acc

                else:
                    n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                    variance=np.var(fitness)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
        variance=np.var(fitness)

    return acc,X,y,fitness,threshold,variance


def execute_heuristic(heuristic,heuristic_param,train_seed,gen,population,acc,X,y,fitness):
    '''Apply the indicated heuristic. This function is called from the function fit (modified by new_fit).'''
    global train_pts_seed
    global last_optimal_evaluations
    global acc_split
    global sample_size,heuristic_freq,last_time_heuristic_accepted
    global unused_bisection_executions, stop_heuristic
    global heuristic_accepted

    threshold=None
    variance=None

    # For the bisection method (sample size, frequency and interpolation).
    df_sample_freq=pd.read_csv('results/data/general/sample_size_freq.csv',index_col=0)
    df_interpolation=pd.read_csv('results/data/SymbolicRegressor/UnderstandingAccuracy/df_Bisection.csv')
    sample_size=int(df_sample_freq[df_sample_freq['env_name']=='SymbolicRegressor']['sample_size'])
    heuristic_freq=int(df_sample_freq[df_sample_freq['env_name']=='SymbolicRegressor']['frequency_time'])
    interpolation_acc=list(df_interpolation['accuracy'])
    interpolation_time=list(df_interpolation['cost_per_eval'])
    lower_time=min(interpolation_time)
    upper_time=max(interpolation_time)

 
    # Set accuracy of the initial generation.
    if gen==0:
        heuristic_accepted=True
        acc,X,y,fitness,threshold,variance=customized_set_initial_accuracy(lower_time,upper_time,list(population),train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=0.95)
        last_time_heuristic_accepted=n_evaluations+n_evaluations_acc
        if heuristic=='II':
            stop_heuristic=False
            unused_bisection_executions=0
        
    # Update accuracy in the rest of the generations.
    else:
        if heuristic=='I':
            acc,X,y,fitness=update_accuracy_heuristicI(acc,lower_time,upper_time,X,y,list(population),train_seed,fitness,heuristic_param,sample_size,heuristic_freq,[interpolation_time,interpolation_acc])
        if heuristic=='II':
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,variance=update_accuracy_heuristicII(acc,lower_time,upper_time,X,y,list(df_seed[5]),list(df_seed[3]),list(population),train_seed,fitness,heuristic_param,sample_size,heuristic_freq,[interpolation_time,interpolation_acc])


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

        # MODIFICATION: count the evaluations made when defining the population.
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
        acc,X,y,fitness,threshold,variance=execute_heuristic(heuristic,heuristic_param,train_seed,gen,population,acc,X,y,fitness)

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
        df_train.append([heuristic_param,train_seed,threshold,variance,heuristic_accepted,acc,gen,score,elapsed_time,generation_time,n_evaluations,n_evaluations_acc,n_evaluations+n_evaluations_acc])
 
        gen+=1# MODIFICATION: update number of generations.

    find_best_individual_final_generation(self,fitness)# MODIFICATION: to obtain the best individual of the last generation.
    
    return self

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# To use the modified fitness function.
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

# validation set and parameters.
test_n_pts=default_train_n_pts
test_pts_seed=1
df_test_pts=build_pts_sample(test_n_pts,test_pts_seed,expr_surf_real)

# Funcion para realizar la ejecucion en paralelo.
def parallel_processing(arg):

    # Extract heuristic information from the function argument.
    heuristic=arg[0]
    param=arg[1]

    # Save training data.
    global df_train
    df_train=[]

    # Initial accuracy.
    init_acc=1/default_train_n_pts# The one corresponding to a set formed by a single point.


    for train_seed in tqdm(list_train_seeds):
        # Training.
        learn(init_acc,train_seed,df_test_pts,heuristic,param)     

    # Save database.
    df_train=pd.DataFrame(df_train,columns=['heuristic_param','train_seed','threshold','variance','update','acc','n_gen','score','elapsed_time','time_gen','n_eval_proc','n_eval_acc','n_eval'])
    df_train.to_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'_param'+str(param)+'.csv')


# Prepare function arguments for parallel processing.
# Argument form: [heuristic identifier, parameter value].
list_arg=[ ['I',0.8],['I',0.95],['II',[10,3]],['II',[5,3]] ]

# Parallel processing.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,list_arg)
pool.close()

# Group databases.
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
                df=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv')
                first=False
            else:
                df_new=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv')
                df=pd.concat([df,df_new],ignore_index=True)


        df.to_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'.csv')


concat_same_heuristic_df(list_arg)



