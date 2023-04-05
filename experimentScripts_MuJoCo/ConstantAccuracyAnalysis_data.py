
# Mediante este script se aplica el algoritmo CMA-ES sobre el entorno Swimmer de MuJoCo, durante 
# un numero maximo de steps, considerando 10 valores diferentes de accuracy y 100 semillas para 
# cada uno de ellos. Por cada valor de accuracy se construira una base de datos con la informacion
# relevante durante en entrenamiento.

#==================================================================================================
# LIBRERIAS
#==================================================================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'cleanrl'))

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
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy
from garage.trainer import TFTrainer

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

import pandas as pd
from dowel.logger import Logger
import warnings
import multiprocessing as mp
import psutil as ps

#==================================================================================================
# NUEVAS FUNCIONES
#==================================================================================================
# FUNCION 1 (validar una politica)
def evaluate_policy(agent,test_env,n_eval_episodes):

    # Fijar semilla del entorno (para que los episodios a validar sean los mismos por cada 
    # llamada a la funcion) y definir el estado inicial (obs) del primer episodio.
    test_env._env.seed(seed=test_env_seed)
    obs,_=test_env.reset()

    # Guardar reward por episodios evaluado.
    all_ep_reward=[]
    for i in range(n_eval_episodes):
        # Evaluar episodio con la politica y guardar el reward asociado.
        episode_reward=0
        done=False
        while not done:
            action,_=agent.get_action(obs)
            env_step_info=test_env.step(action)
            episode_reward+=env_step_info.reward
            done=env_step_info.last
            obs=env_step_info.observation
        all_ep_reward.append(episode_reward)
        obs,_=test_env.reset()

    return np.mean(all_ep_reward)

# FUNCION 2 (aprender politica optima con CMA-ES)
@wrap_experiment(snapshot_mode="none", log_dir="/tmp/",archive_launch_repo=False)
def learn(ctxt=None, gymEnvName=None, action_space=None, max_episode_length=None,
                       policy_name=None,seed=None,accuracy=1.0):

    # Inicializacion de contadores.
    global n_steps,n_episodes,n_generations
    n_steps=0 # Contador que indica el numero de steps consumidos hasta el momento.
    n_episodes = 0 # Contador que indica en que episodio estamos.
    n_generations=0 # Contador que indica en que generacion estamos.

    # Definicion de parametros para el algoritmos CMA-ES.
    global popsize,batch_size_ep,total_generations
    popsize= 20 # Tamaño de las generaciones/poblaciones en CMA-ES. (Referencia: https://github.com/rlworkgroup/garage/blob/master/src/garage/examples/np/cma_es_cartpole.py)
    batch_size_ep=1 # Numero de episodios que se van a considerar para evaluar cada politica/individuos de una generacion.

    # Fijar semilla.
    set_seed(seed)

    with TFTrainer(ctxt) as trainer:

        # Definir entorno de entrenamiento con el accuracy seleccionado.
        global current_max_episode_length
        current_max_episode_length=max_episode_length*accuracy
        train_env = GymEnv(gymEnvName, max_episode_length=current_max_episode_length)
        train_env._env.unwrapped.model.opt.timestep=default_frametime/accuracy
        train_env._env.seed(seed=train_env_seed)

        # Definir entorno de validacion con el accuracy maximo.
        global test_env
        test_env = GymEnv(gymEnvName, max_episode_length=max_episode_length)
        test_env._env.unwrapped.model.opt.timestep=default_frametime
        
        # Definir politica.
        is_action_space_discrete=bool(["continuous", "discrete"].index(action_space))
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name=policy_name, env_spec=train_env.spec)
        else:
            policy = ContinuousMLPPolicy(name=policy_name, env_spec=train_env.spec)
        sampler = LocalSampler(agents=policy, envs=train_env, max_episode_length=train_env.spec.max_episode_length, is_tf_worker=True)

        # Inicializar algoritmo CMA-ES.
        algo= CMAES(env_spec=train_env.spec, policy=policy, sampler=sampler, n_samples=popsize)
        trainer.setup(algo, train_env)

        # Entrenamiento.
        total_generations= int(max_steps/(popsize*batch_size_ep*current_max_episode_length)) # Numero de generaciones a evaluar, dependiendo del numero maximo de steps definido para entrenar. (CRITERIO DE PARADA)
        trainer.train(n_epochs=total_generations, batch_size=batch_size_ep*current_max_episode_length)


#==================================================================================================
# FUNCIONES DISEÑADAS PARA SUSTITUIR ALGUNAS YA EXISTENTES
#==================================================================================================
# FUNCION 3 (para poder evaluar cada politica de cada generacion durante el entrenamiento con el mismo
# conjunto de episodios, esta modificacion hace que el proceso sea determinista y las comparaciones 
# entre individuos por generacion sean justas).
def start_episode(self):
    """Begin a new episode."""

    self._eps_length = 0

    # MODIFICACION: para que siempre se usen los mismos episodios para validar las politicas/individuos y estas puedan ser comparables.
    global n_episodes,batch_size_ep,n_generations
    if n_episodes%batch_size_ep==0:
        self.env._env.seed(seed=n_generations)#seed=train_env_seed (si se quiere evaluar los mismos episodios en todas las generaciones)
    self._prev_obs, episode_info = self.env.reset()

    for k, v in episode_info.items():
        self._episode_infos[k].append(v)

    self.agent.reset()

# FUNCION 4 (para poder almacenar los datos de interes durante el entrenamiento).
def rollout(self):
    global n_steps,n_episodes,n_generations,df_acc,current_max_episode_length,n_eval_episodes,popsize,total_generations
    global seed,accuracy
    global list_gen_policies,list_gen_policies_rewards,policy_reward_per_ep

    # Listas para guardar las politicas asociadas a una generacion y sus correspondientes rewards.
    n_policies=n_episodes/batch_size_ep #Politicas evaluadas hasta el momento.
    if n_policies%popsize==0:

        list_gen_policies=[]
        list_gen_policies_rewards=[]

        policy_reward_per_ep=[]

    # En caso de usar un entorno con criterio de parada dependiente del parametro 
    # terminate_when_unhealthy, anular este tipo de parada.
    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    # Inicializar episodio.
    self.start_episode() 
    self._max_episode_length = current_max_episode_length # Fijar tamaño maximo del episodio.
    
    # Actualizar contadores tras evaluar el episodio.
    episode_steps = 0 # Inicializar contador para sumar el numero de steps dados en este episodio.
    policy_reward=0
    while not self.step_episode(): # Hasta que no se puedan dar mas steps en el episodio.
        policy_reward+=self._env_steps[episode_steps].reward # Dar un nuevo step.
        episode_steps+= 1 # Sumar step.

    n_steps += episode_steps #  Steps totales dados hasta el momento.
    n_episodes += 1 # Episodios evaluados hasta el momento.
    policy_reward_per_ep.append(policy_reward)

    # Cuando se haya evaluado una politica/individuo al completo (numero de episodios evaluado=batch_size_ep),
    # se almacena la politica y su correspondiente reward (reward total en batch_size_ep).
    if n_episodes%batch_size_ep==0:
        list_gen_policies.append(self.agent)
        list_gen_policies_rewards.append(np.mean(policy_reward_per_ep))
        policy_reward_per_ep=[]
    
    # Cuando se han evaluado las suficientes politicas como para poder completar una generacion nueva,
    # se guarda la informacion obtenida de la nueva generacion.
    n_policies=n_episodes/batch_size_ep#Politicas evaluadas hasta el momento.
    if n_policies!=0 and n_policies%popsize == 0:
        best_policy=list_gen_policies[list_gen_policies_rewards.index(max(list_gen_policies_rewards))]
        reward=evaluate_policy(best_policy,test_env,n_eval_episodes)
        df_acc.append([accuracy,seed,n_generations,n_episodes,n_steps,reward])
        n_generations+=1

        # print('GENERACION: '+str(n_generations)+'/'+str(total_generations))

    return self.collect_episode()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Modificacion de funcion existente (para guardar datos durante el entrenamiento).
garage.sampler.default_worker.DefaultWorker.rollout = rollout
# Modificacion de funcion existente ( para no perder tiempo).
garage.trainer.Trainer.save = lambda self, epoch: "skipp save."
# Modificacion de funcion existente (para no imprimir informacion de episodio durante el entrenamiento).
Logger.log= lambda self, data: 'skipp info message.'
warnings.filterwarnings("ignore")
# Modificacion de funcion existente (para evaluar todas las politicas/individuos de cada generacion con los mismos episodios y asi puedan ser comparables).
from garage.sampler.default_worker import DefaultWorker
DefaultWorker.start_episode=start_episode

# Caracteristicas del entorno.
gymEnvName='Swimmer-v3'
action_space="continuous"
max_episode_length=1000
default_frametime=0.01 # Parametro del que se modificara el accuracy.
policy_name='SwimmerPolicy'
DTU=False # Criterio de parada dependiente de terminate_when_unhealthy.

# Mallados y parametros para el entrenamiento.
list_train_seeds = list(range(2,102,1)) # Lista de semillas de entrenamiento.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
train_env_seed=0 # Semilla para el entorno de entrenamiento.
max_steps=1000000 # Limite de entrenamiento medido en steps (asumiendo que el coste de todos los steps es el mismo). (Referencia: https://huggingface.co/sb3/ppo-Swimmer-v3/tree/main)

# Parametros de validacion.
test_env_seed=1 # Semilla para el entorno de validacion.
n_eval_episodes=10 # Numero de episodios que se evaluaran en la validacion.

# Funcion para ejecucion en paralelo.
def parallel_processing(arg):
    global df_acc
    df_acc=[]

    global seed,accuracy
    accuracy=arg
    for seed in tqdm(list_train_seeds):
        learn(gymEnvName=gymEnvName, action_space=action_space, max_episode_length=max_episode_length,policy_name=policy_name,seed=seed,accuracy=accuracy) 

    # Guardar base de datos.
    df_acc=pd.DataFrame(df_acc,columns=['accuracy','train_seed','n_gen','n_ep','n_steps','reward'])
    df_acc.to_csv('results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc'+str(accuracy)+'.csv')

# Procesamiento en paralelo.
phisical_cpu=ps.cpu_count(logical=True)
pool=mp.Pool(phisical_cpu)
pool.map(parallel_processing,list_acc)
pool.close()

# Guardar lista de accuracys.
np.save('results/data/MuJoCo/ConstantAccuracyAnalysis/list_acc',list_acc)

# Guardar limite de entrenamiento.
np.save('results/data/MuJoCo/ConstantAccuracyAnalysis/max_steps',max_steps)








