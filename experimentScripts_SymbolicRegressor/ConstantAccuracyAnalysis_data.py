#==================================================================================================
# LIBRERÍAS
#==================================================================================================
# Para mi código.
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pandas as pd
from tqdm import tqdm

# Para las modificaciones.
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
# NUEVAS FUNCIONES
#==================================================================================================

def mean_abs_err(z_test,z_pred):
    return sum(abs(z_test-z_pred))/len(z_test)

def build_pts_sample(n_sample,seed):

    # Fijar la semilla.
    rng = check_random_state(0)

    # Mallado aleatorio (x,y).
    xy_sample=rng.uniform(-1, 1, n_sample*2).reshape(n_sample, 2)

    # Calcular alturas correspondientes (valor z).
    expr_surf=random_surface(seed, xy_sample.shape[1])
    z_sample=expr_surf.execute(xy_sample)

    # Todos los datos en un array.
    pts_sample=np.insert(xy_sample, xy_sample.shape[1], z_sample, 1)

    return pts_sample,expr_surf

def split_pts_train_test(df_pts,test_n_pts):

    # Todos los indices de las filas.
    ind_rows=range(0,df_pts.shape[0])

    # Seleccionar los puntos de validación y de entrenamiento.
    ind_test_rows=np.random.choice(ind_rows,test_n_pts,replace=False)
    df_test_pts=df_pts[ind_test_rows,]
    df_train_pts=np.delete(df_pts, ind_test_rows, axis=0)

    return df_train_pts,df_test_pts

def evaluate(df_test_pts,est_surf):

    # Dividir base de datos con las coordenadas de los puntos.
    xy_test=df_test_pts[:,[0,1]]
    z_test=df_test_pts[:,2]

    # Calcular el valor de las terceras coordenadas con las superficie seleccionada.
    z_pred=est_surf.predict(xy_test)

    # Calcular score asociado al conjunto de puntos para la superficie seleccionada.
    score=mean_abs_err(z_test, z_pred)

    return score   

def learn(df_train_pts,train_seed,df_test_pts,max_time):

	# Definición del algoritmo genético con el cual se encontrarán la superficie.
    est_surf=SymbolicRegressor(population_size=1000,
                               function_set= ('add', 'sub', 'mul', 'div','log','sin','cos','tan'),
                               verbose=0, random_state=0)

	# Ajustar la superficie a los puntos.
    xy_train=df_train_pts[:,[0,1]]
    z_train=df_train_pts[:,2]
    est_surf.fit(xy_train, z_train,max_time,train_seed,df_test_pts)    

    return est_surf._program 

def draw_surface_pts(df_train_pts,df_test_pts,expr_surf_real,expr_surf_pred,accuracy,seed):

    ax = plt.subplot(111, projection='3d')

    # Superficie real.
    x= np.arange(-1, 1, 0.1)
    y= np.arange(-1, 1, 0.1)
    x,y=np.meshgrid(x, y)
    z=[]
    for i in range(len(x)):
        xy=np.transpose(np.array([x[i],y[i]]))
        z.append(np.array(expr_surf_real.execute(xy)))
    z=np.array(z)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='green', alpha=0.5)

    # Superficie predicha.
    z=[]
    for i in range(len(x)):
        xy=np.transpose(np.array([x[i],y[i]]))
        z.append(np.array(expr_surf_pred.execute(xy)))
    z=np.array(z)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellow', alpha=0.5)

    # Puntos de entrenamiento.
    ax.scatter(df_train_pts[:,0],df_train_pts[:,1], df_train_pts[:,2],c='blue',alpha=1,s=2,label='Train pts')

    # Puntos de validación.
    ax.scatter(df_test_pts[:,0],df_test_pts[:,1], df_test_pts[:,2],c='red',alpha=1,s=2,label='Test pts')

    ax.set_title('Accuracy:'+str(accuracy)+'; Seed:'+str(seed))
    ax.legend()
    plt.show()

#==================================================================================================
# FUNCIONES DISEÑADAS A PARTIR DE ALGUNAS YA EXISTENTES
#==================================================================================================

def find_best_individual_final_generation(self,fitness):

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

def new_fit(self, X, y, max_time, train_seed,df_test_pts,sample_weight=None):
    """Fit the Genetic Program according to X, y.

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

    ###############################################################################################
    #MODIFICACIONES: 
    start_total_time=time() #AQUÍ: empezar a contar el tiempo de entrenamiento
    gen=0# AQUÍ: para que el procedimiento no termine cuando se alcance un número de generaciones, las generaciones se cuentan con un contador independiente.

    while time()-start_total_time < max_time: #AQUÍ: El procedimiento terminará al sobrepasar el tiempo predefinido

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
        else:
            best_fitness = fitness[np.argmin(fitness)]
          

        find_best_individual_final_generation(self,fitness) #AQUÍ: para poder evaluar la mejor superficie durante el proceso.

        #AQUÍ: ir guardando los datos de interés durante el entrenamiento. 
        score=evaluate(df_test_pts,self)
        elapsed_time=time()-start_total_time      
        df_train.append([train_seed,gen,score,elapsed_time,generation_time])
        
        gen+=1# AQUÍ: actualizar número de generaciones.
    find_best_individual_final_generation(self,fitness)#AQUÍ: para obtener el mejor elemento de la última generación.
    ###############################################################################################
    return self

def random_surface(seed, n_features):
    # Para crear una función simbólica con la semilla y las funciones simples que se deseen.
    self=SymbolicRegressor(population_size=1,
                           function_set= ('add', 'sub', 'mul', 'div','log','sin','cos','tan'),
                           verbose=0, random_state=seed)

    #params
    params = self.get_params()

    self._function_set = []
    for function in self.function_set:
        if isinstance(function, str):
            self._function_set.append(_function_map[function])
        elif isinstance(function, _Function):
            self._function_set.append(function)

    self._arities = {}
    for function in self._function_set:
        arity = function.arity
        self._arities[arity] = self._arities.get(arity, [])
        self._arities[arity].append(function)

    if hasattr(self, '_transformer'):
        params['_transformer'] = self.transformer
    else:
        params['_transformer'] = None

    #program
    program = _Program(function_set=self._function_set,
                        arities=self._arities,
                        init_depth=params['init_depth'],
                        init_method=params['init_method'],
                        n_features=n_features,
                        metric=self.metric,
                        transformer=params['_transformer'],
                        const_range=params['const_range'],
                        p_point_replace=params['p_point_replace'],
                        parsimony_coefficient=params['parsimony_coefficient'],
                        feature_names=params['feature_names'],
                        random_state=check_random_state(seed),
                        program=None)

    program.build_program(check_random_state(seed))

    return program
    
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Para usar la función de ajuste modificada.
BaseSymbolic.fit=new_fit

# Cardinal de conjunto inicial predefinido.
default_train_n_pts=50

# Mallados.
list_seeds=range(1,31,1)
list_acc=[1.0,0.8,0.6,0.4,0.2]

# Parámetros de entrenamiento.
max_time=60*2

# Parámetros de validación.
test_n_pts=default_train_n_pts

# Construir tabla con datos de entrenamiento.
for accuracy in list_acc:

    # Cambiar cardinal predefinido.
    train_n_pts=int(default_train_n_pts*accuracy)

    # Guardar datos de entrenamiento.
    df_train=[]
    for seed in tqdm(list_seeds):

        # Crear conjuntos generales de entrenamiento y validación.
        total_n_pts=test_n_pts+train_n_pts
        df_pts,expr_surf_real=build_pts_sample(total_n_pts,seed)
        print(df_pts)
        df_train_pts,df_test_pts=split_pts_train_test(df_pts,test_n_pts)

        # Entrenamiento.
        expr_surf_pred=learn(df_train_pts,seed,df_test_pts,max_time)

        # Dibujar puntos de entrenamiento y validación junto a superficies.
        #draw_surface_pts(df_train_pts,df_test_pts,expr_surf_real,expr_surf_pred,accuracy,seed)
      

    df_train=pd.DataFrame(df_train,columns=['train_seed','n_gen','score','elapsed_time','time_gen'])
    df_train.to_csv('results/data/SymbolicRegressor/df_train_acc'+str(accuracy)+'_.csv')

# Guardar lista de precisiones.
np.save('results/data/SymbolicRegressor/list_acc',list_acc)

