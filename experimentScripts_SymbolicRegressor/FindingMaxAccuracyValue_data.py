'''
It is intended to find the number of training points from which the MAE of the surface selected 
by the GP starts to converge, considering different original surfaces (to which the original set 
of points belongs). This script builds a database for each original surface considered. It stores 
the scores of the optimal surfaces chosen by the GP from a set of points of different sizes 
defined with different seeds.

Based on: https://github.com/trevorstephens/gplearn/blob/main/doc/gp_examples.ipynb
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
from tqdm import tqdm as tqdm

# for modifications made to existing codes.
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

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def mean_abs_err(z_test,z_pred):
    '''Calculate the mean absolute error (MAE) between two vectors.'''
    return sum(abs(z_test-z_pred))/len(z_test)

def build_pts_sample(n_sample,seed,eval_expr):
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
    z_sample=eval(eval_expr)

    # All data in a single array.
    pts_sample=np.insert(xy_sample, xy_sample.shape[1], z_sample, 1)

    return pts_sample

def learn(df_train_pts,max_time):
    '''
    Execute the GP with a given train point set and during a given maximum time.

    Parameters
    ==========
    df_test_pts: Database with the three coordinates of the points that form the training set. 
    max_time: Maximum time set for the GP execution (surface search).

    Return
    ======
    Surface selected as optimal.
    '''

	# Definition of the genetic algorithm with which the surface will be found.
    est_surf = SymbolicRegressor(random_state=0)

	# Adjust the surface to the points.
    xy_train=df_train_pts[:,[0,1]]
    z_train=df_train_pts[:,2]
    est_surf.fit(xy_train, z_train,max_time)      

    return est_surf

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
    score_mae=mean_abs_err(z_test,z_pred)

    return score_mae

def find_max_accuracy(list_train_n_pts,list_train_seeds,test_n_pts,test_seed,eval_expr,max_time):
    '''
    Build a database with the information obtained for each GP run using sets of points of different sizes
    selected using different seeds.
    
    Parameters
    ==========
    list_train_n_pts: List with the sizes considered for the set of training points.
    list_train_seeds: List of seeds for the random selection of the points that form the training set. 
    test_n_pts: Number of points that will form the validation set.
    test_seed: Seed to randomly select the points that will form the validation set.
    eval_expr: Expression of the original surface.
    max_time: Maximum time set for GP execution (surface search).

    Return
    ======
    Database with the scores associated with different training set sizes, defined from different seeds.
    '''

    # Build a set of validation points.
    df_test_pts=build_pts_sample(test_n_pts,test_seed,eval_expr)

    # Store data of interest in a database.
    df=[]

    for n_pts in tqdm(list_train_n_pts):
        for seed in list_train_seeds:
            # Build the sample of points to train.
            df_train_pts=build_pts_sample(n_pts,seed,eval_expr)
            # Learn surface.
            est_surf=learn(df_train_pts,max_time)
            # Validation.
            score_mae=evaluate(df_test_pts,est_surf)
            # Save data.
            df.append([n_pts,seed,score_mae])

    return pd.DataFrame(df,columns=['n_pts','seed','score_mae'])

def new_fit(self, X, y, max_time, sample_weight=None):
    """
    This function replaces the existing fit function.
    
    Fit the Genetic Program according to X, y.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    sample_weight : array-like, shape = [n_samples], optional
        Weights applied to individual samples.

    Returns
    -------
    self : object
        Returns self.

    """
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
                                'pearson', 'spearman'):
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]
    elif isinstance(self, ClassifierMixin):
        if self.metric != 'log loss':
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]
    elif isinstance(self, TransformerMixin):
        if self.metric not in ('pearson', 'spearman'):
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

    
    start_total_time=time() # MODIFICATION: start counting training time.
    gen=0
    while time()-start_total_time < max_time:# MODIFICATION: so that the process ends when the set time is reached.

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
            # if best_fitness >= self.stopping_criteria:
            #     break
        else:
            best_fitness = fitness[np.argmin(fitness)]
            # if best_fitness <= self.stopping_criteria:
            #     break
        
		# MODIFICATION: modify the stop criteria according to the training time.
        total_time=time()-start_total_time
        if total_time>=max_time:
            break

        gen+=1 # MODIFICATION: manually update the number of generations.
        

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

    return self


#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# To use the modified adjustment function.
BaseSymbolic.fit=new_fit

#--------------------------------------------------------------------------------------------------
# EXAMPLE 1 (Hyperbolic paraboloid)
#--------------------------------------------------------------------------------------------------

# Parameters.
list_train_n_pts=range(2,32,2)
list_train_seeds=range(1,21,1)
test_n_pts=30
test_seed=0
eval_expr1='x**2-y**2+y-1'
max_time=30

# Save data.
df1=find_max_accuracy(list_train_n_pts,list_train_seeds,test_n_pts,test_seed,eval_expr1,max_time)
df1.to_csv('results/data/SymbolicRegressor/FindingMaxAccuracyValue/FindingMaxAccuracyValue1.csv')
np.save('results/data/SymbolicRegressor/FindingMaxAccuracyValue/eval_expr1',eval_expr1)

#--------------------------------------------------------------------------------------------------
# EXAMPLE 2 (Plane)
#--------------------------------------------------------------------------------------------------
# Parameters.
list_train_n_pts=range(2,7,1)
list_train_seeds=range(1,21,1)
test_n_pts=30
test_seed=0
eval_expr2='x+2*y+3'
max_time=30

# Save data.
df2=find_max_accuracy(list_train_n_pts,list_train_seeds,test_n_pts,test_seed,eval_expr2,max_time)
df2.to_csv('results/data/SymbolicRegressor/FindingMaxAccuracyValue/FindingMaxAccuracyValue2.csv')
np.save('results/data/SymbolicRegressor/FindingMaxAccuracyValue/eval_expr2',eval_expr2)





