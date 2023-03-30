#==================================================================================================
# LIBRERIAS
#==================================================================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'


from argparse import ArgumentError
from statistics import median
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import subprocess
import time
import re
from os.path import exists
import sys
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse

from garage import wrap_experiment
import garage.sampler.default_worker
from garage.sampler.default_worker import *
from garage._functions import *
import numpy as np

import garage.trainer
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.algos import CMAES
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy, ContinuousMLPPolicy
from garage.trainer import TFTrainer

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

#==================================================================================================
# FUNCIONES
#==================================================================================================
def rollout(self):
    global start_global_time
    global total_steps
    global n_episode
    global list_gen_times_per_episode
    global res_filepath
    global max_episode_length

    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    if start_global_time is None:
        start_global_time = time.time()

    # Inicializar contadores para el episodio.
    i = -1 # Inicializar contador para sumar el número de steps dados en este episodio.
    sum_of_rewards = 0 # Para sumar el número de rewards en el actual episodio.
    episode_start_ref_t = time.time() # Inicializar contador para medir tiempo de este episodio.

    # Inicializar episodio.
    self.start_episode() 
    self._max_episode_length = max_episode_length # Fijar tamaño máximo del episodio.

    # Actualizar contadores propios del episodio.
    while not self.step_episode(): # Hasta que no se puedan dar más steps en el episodio.
        i += 1 # Sumar step.
        sum_of_rewards += self._env_steps[i-1].reward # Sumar reward asociado al step.
    episode_runtime=time.time() - episode_start_ref_t

    # Actualizar contadores generales.
    total_steps += i #  Steps totales dados hasta el momento.
    list_gen_times_per_episode.append(episode_runtime) # Tiempo de ejecución gastado hasta el momento.
    n_episode += 1 # Episodios evaluados hasta el moemento.
    
    # Cuando se han evaluado los  suficientes episodios como para poder completar una generación nueva,
    # se guarda la información obtenida de la nueva generación en una nueva línea del fichero "res_filepath".
    if n_episode%popsize == popsize-1:
        runtimes = "("+";".join(map(str, list_gen_times_per_episode))+")"
        with open(res_filepath, "a+") as f:
            print("seed_"+str(seed)+"_gymEnvName_"+gymEnvName, time.time() - start_global_time, total_steps, n_episode, runtimes , file=f, sep=",", end="\n")
        list_gen_times_per_episode = []

    return self.collect_episode()

@wrap_experiment(snapshot_mode="none", log_dir="/tmp/",archive_launch_repo=False)
def launch_from_python(ctxt=None,seed=None, gymEnvName=None, action_space=None, max_episode_length=None,index=None):
    global DTU

    if "DTU" in gymEnvName:
        DTU = True
    else:
        DTU= False

    is_action_space_discrete=bool(["continuous", "discrete"].index(action_space))

    set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        # Definir entorno.
        if DTU: 
            env = GymEnv(gymEnvName.replace("_DTU", ""), max_episode_length=max_episode_length)
        else:
            env = GymEnv(gymEnvName, max_episode_length=max_episode_length)
            
        # Definir política.
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name=str(index), env_spec=env.spec)
        else:
            policy = ContinuousMLPPolicy(name=str(index), env_spec=env.spec)
        sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True)

        # Inicializar algoritmo CMA-ES.
        algo= CMAES(env_spec=env.spec, policy=policy, sampler=sampler, n_samples=popsize)
        trainer.setup(algo, env)

        # Entrenamiento.
        trainer.train(n_epochs=n_generations, batch_size=batch_size)
    
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Modificación de función existente para introducir criterio de parada temprano.
garage.sampler.default_worker.DefaultWorker.rollout = rollout
# Modificación de función existente para no perder tiempo.
garage.trainer.Trainer.save = lambda self, epoch: print("skipp save.")


# Parámetros para CMA-ES.
popsize = 10  # Tamaño de las generaciones poblaciones en CMA-ES.
n_generations = 5# Número de generaciones.
list_train_seeds = list(range(2,3))

# Otras variables.
batch_size=1 # Cada cuantos steps se acumula el aprendizaje.
list_gen_times_per_episode = [] # Lista para almacenar los tiempos de ejecución por episodio asociados a una generación. 
start_global_time = None # Variable que marca el tiempo de inicio de un proceso de entrenamiento.
total_steps = 0 # Contador de número de steps total.
n_episode = 0 # Contador que indica en que episodio estamos.


# Parámetros asociados a los entornos MuJoCo
gymEnvName_list =         ['CartPole-v1',  'Pendulum-v1',  'HalfCheetah-v3',  'InvertedDoublePendulum-v2',  'Swimmer-v3', 'Hopper-v3' , 'Ant-v3'    , 'Walker2d-v3', 'Hopper-v3_DTU' , 'Ant-v3_DTU'    , 'Walker2d-v3_DTU']
action_space_list =       ["discrete"   ,  "continuous",  "continuous"    ,  "continuous"               ,  "continuous", "continuous", "continuous", "continuous" , "continuous"    , "continuous"    , "continuous"     ]
max_episode_length_list = [          400,            200,  1000            ,                         1000,  1000        ,         1000,    1000     ,  1000        ,         1000    ,    1000         ,  1000            ]
is_reward_monotone_list = [True         ,  True         ,  False           ,  False                      ,   False      , False       ,  False      , False        , False           ,  False          , False            ]


for index,gymEnvName, action_space, max_episode_length, is_reward_monotone in zip(range(len(gymEnvName_list)),gymEnvName_list, action_space_list, max_episode_length_list,  is_reward_monotone_list):

    print('ENVIRONMENT: '+str(gymEnvName))
    for seed in list_train_seeds:
        res_filepath = f"others/demoScripts/results/mujoco_CMA_ES/gymEnvName_{gymEnvName}_{seed}.txt"
        launch_from_python(seed=seed, gymEnvName=gymEnvName, action_space=action_space, max_episode_length=max_episode_length,index=index) 

