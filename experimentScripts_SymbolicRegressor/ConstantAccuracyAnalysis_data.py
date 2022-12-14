# Mediante este script se analiza la influencia de considerar conjuntos de puntos de diferentes 
# tamaños a la hora de entrenar/buscar la superficie a la que pertenecen, usando una regresión 
# simbólica con GA. Para ello, se escoge la superficie x²-y²+y-1 y se consideran 7 tamaños/precisiones
# para el conjunto de puntos inicial sobre el cual se entrenará/buscará la superficie. Para cada
# precisión se repite el procedimiento usando 30 semillas diferentes (esto permite definir la población
# inicial del GA de forma diferente). Para la búsqueda de las superficies se fijará un número máximo
# de evaluaciones (en la expresión de la superficie seleccionada) y se guardarán los datos relevantes 
# asociados a la mejor superficie encontrada durante el proceso de búsqueda.

# Basado en: https://github.com/trevorstephens/gplearn/blob/main/doc/gp_examples.ipynb

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

# FUNCIÓN 1
# Parámetros:
#   >z_test: terceras coordenadas reales de los puntos de la superficie.
#   >z_pred: terceras coordenadas obtenidas a partir de la superficie predicha.
# Devuelve: el error absoluto medio de las dos listas anteriores.

def mean_abs_err(z_test,z_pred):
    return sum(abs(z_test-z_pred))/len(z_test)

# FUNCIÓN 2
# Parámetros:
#   >n_sample: número de puntos que se desean construir.
#   >seed: semilla para la selección aleatoria de los puntos.
#   >eval_expr: expresión de la superficie de la cual se quiere extraer la muestra de puntos.
# Devuelve: base de datos con las tres coordenadas de los puntos de la muestra.

def build_pts_sample(n_sample,seed,expr_surf):

    # Fijar la semilla.
    rng = check_random_state(seed)

    # Mallado aleatorio (x,y).
    xy_sample=rng.uniform(-1, 1, n_sample*2).reshape(n_sample, 2)
    x=xy_sample[:,0]
    y=xy_sample[:,1]

    # Calcular alturas correspondientes (valor z).
    z_sample=eval(expr_surf)

    # Todos los datos en un array.
    pts_sample=np.insert(xy_sample, xy_sample.shape[1], z_sample, 1)

    return pts_sample

# FUNCIÓN 3
# Parámetros:
#   >df_test_pts: base de datos con las tres coordenadas de los puntos que forman el 
#    conjunto de validación.
#   >est_surf: superficie seleccionada en el proceso GA de entrenamiento.
# Devuelve: error absoluto medio.

def evaluate(df_test_pts,est_surf):

    # Dividir base de datos con las coordenadas de los puntos.
    xy_test=df_test_pts[:,[0,1]]
    z_test=df_test_pts[:,2]

    # Calcular el valor de las terceras coordenadas con las superficie seleccionada.
    z_pred=est_surf.predict(xy_test)

    # Calcular score asociado al conjunto de puntos para la superficie seleccionada.
    score=mean_abs_err(z_test, z_pred)

    return score   

# FUNCIÓN 4
# Parámetros:
#   >df_train_pts: base de datos con las tres coordenadas de los puntos que forman el 
#    conjunto de entrenamiento.
#   >train_seed: semilla de entrenamiento.
#   >df_test_pts: base de datos con las tres coordenadas de los puntos que forman el 
#    conjunto de validación.
#   >max_time: tiempo máximo fijado para la ejecución del GA (búsqueda de la superficie).
# Devuelve: superficie seleccionada.

def learn(df_train_pts,train_seed,df_test_pts,max_time):

	# Definición del algoritmo genético con el cual se encontrarán la superficie.
    est_surf=SymbolicRegressor(function_set= ('add', 'sub', 'mul', 'div','sqrt','log','abs','neg','inv','max','min','sin','cos','tan'),
                               verbose=0, random_state=train_seed)

	# Ajustar la superficie a los puntos.
    xy_train=df_train_pts[:,[0,1]]
    z_train=df_train_pts[:,2]
    est_surf.fit(xy_train, z_train,max_time,train_seed,df_test_pts)    

    return est_surf._program 


#==================================================================================================
# FUNCIONES DISEÑADAS A PARTIR DE ALGUNAS YA EXISTENTES
#==================================================================================================

# FUNCIÓN 5
# -Original: raw_fitness
# -Script: _Program.py
# -Clase: _Program
def new_raw_fitness(self, X, y, sample_weight):
    
    y_pred = self.execute(X)
    if self.transformer:
        y_pred = self.transformer(y_pred)
    raw_fitness = self.metric(y, y_pred, sample_weight)
    
    # MODIFICACIÓN: Sumar el número de evaluaciones realizadas (tantas como puntos en el 
    # conjunto de entrenamiento).
    global n_evaluations
    n_evaluations+=X.shape[0]

    return raw_fitness

# FUNCIÓN 6
# -Original: _parallel_evolve
# -Script: genetic.py

def new_parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
   
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
    i=0# MODIFICACIÓN: inicializar contador de forma manual.
    while i<n_programs and n_evaluations<max_n_eval:#MODIFICACIÓN: añadir nueva restricción para terminar el bucle.

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

        i+=1# MODIFICACIÓN: actualizar contador de forma manual.
    return programs

# FUNCIÓN 7
# Esta función contiene una parte del código interno de una función ya existente.
# -Original: fit
# -Script: genetic.py 
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

# FUNCIÓN 8
# -Original: fit
# -Script: genetic.py
# -Clase: BaseSymbolic
def new_fit(self, X, y, max_n_eval, train_seed,df_test_pts,sample_weight=None):# MODIFICACIÓN: añadir nuevos argumentos.

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

    start_total_time=time() #MODIFICACIÓN: empezar a contar el tiempo de entrenamiento.
    gen=0# MODIFICACIÓN: para que el procedimiento no termine cuando se alcance un número de generaciones, las generaciones se cuentan con un contador independiente.
    
    # MODIFICACIÓN: variable global mediante la cual se irán contando el número de evaluaciones realizadas,
    # entendiendo por evaluación cada evaluación de un punto en una expresión de una superficie.
    global n_evaluations
    n_evaluations=0

    while n_evaluations < max_n_eval:# MODIFICACIÓN: modificar el límite de entrenamiento.

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
        

        find_best_individual_final_generation(self,fitness) # MODIFICACIÓN: para poder evaluar la mejor superficie durante el proceso.
        
        # MODIFICACIÓN: ir guardando los datos de interés durante el entrenamiento. 
        score=evaluate(df_test_pts,self)
        elapsed_time=time()-start_total_time      
        df_train.append([train_seed,gen,score,elapsed_time,generation_time,n_evaluations])
        
        gen+=1# MODIFICACIÓN: actualizar número de generaciones.

    find_best_individual_final_generation(self,fitness)# MODIFICACIÓN: para obtener el mejor individuo de la última generación.
    
    return self

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Para usar la función de ajuste modificada.
_Program.raw_fitness=new_raw_fitness
_parallel_evolve=new_parallel_evolve
BaseSymbolic.fit=new_fit

# Superficie.
expr_surf_real='x**2-y**2+y-1'

# Mallados.
list_train_seeds=range(1,31,1)
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4]

# Parámetros de entrenamiento.
default_train_n_pts=50# Cardinal de conjunto inicial predefinido.
train_pts_seed=0
max_n_eval=10000000

# Parámetros y conjunto de validación.
test_n_pts=default_train_n_pts
test_pts_seed=1
df_test_pts=build_pts_sample(test_n_pts,test_pts_seed,expr_surf_real)

# Construir tabla con datos de validación.
for accuracy in list_acc:

    # Cambiar cardinal predefinido.
    train_n_pts=int(default_train_n_pts*accuracy)

    # Crear conjunto de entrenamiento.
    df_train_pts=build_pts_sample(train_n_pts,train_pts_seed,expr_surf_real)

    # Guardar datos de entrenamiento.
    df_train=[]
    for train_seed in tqdm(list_train_seeds):

        # Entrenamiento.
        expr_surf_pred=learn(df_train_pts,train_seed,df_test_pts,max_n_eval)

    df_train=pd.DataFrame(df_train,columns=['train_seed','n_gen','score','elapsed_time','time_gen','n_eval'])
    df_train.to_csv('results/data/SymbolicRegressor/df_train_acc'+str(accuracy)+'.csv')

# Guardar lista de precisiones.
np.save('results/data/SymbolicRegressor/list_acc',list_acc)

# Guardar expresión de superficie.
np.save('results/data/SymbolicRegressor/expr_surf',expr_surf_real)

