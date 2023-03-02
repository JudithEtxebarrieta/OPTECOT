#==================================================================================================
# LIBRERÍAS
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
from garage.trainer import TFTrainer,Trainer

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

import pandas as pd
from dowel.logger import Logger
import warnings
import multiprocessing as mp
import psutil as ps

import scipy as sc
import random
import cma

#==================================================================================================
# NUEVAS FUNCIONES
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Funciones para el proceso de aprendizaje o búsqueda de la óptima política.
#--------------------------------------------------------------------------------------------------
# FUNCIÓN (evaluar una política)
def evaluate_policy(agent,env,n_eval_episodes,type_eval_test=True):
    global n_generations

    # Definir semilla.
    if type_eval_test:
        env_seed=test_env_seed
    else:
        env_seed=n_generations

    # Fijar semilla del entorno (para que los episodios a validar sena los mismos por cada 
    # llamada a la función) y definir el estado inicial (obs) del primer episodio.
    env._env.seed(seed=env_seed)
    obs,_=env.reset()

    # Guardar reward por episodios evaluado.
    all_ep_reward=[]
    for _ in range(n_eval_episodes):
        # Evaluar episodio con la política y guardar el reward asociado.
        episode_reward=0
        done=False
        while not done:
            action,_=agent.get_action(obs)
            env_step_info=env.step(action)
            episode_reward+=env_step_info.reward
            done=env_step_info.last
            obs=env_step_info.observation
        all_ep_reward.append(episode_reward)
        obs,_=env.reset()

    return np.mean(all_ep_reward)

# FUNCIÓN  (aprender política óptima con CMA-ES)
@wrap_experiment(snapshot_mode="none", log_dir="/tmp/",archive_launch_repo=False)
def learn(ctxt=None, gymEnvName=None, action_space=None, max_episode_length=None,
                       policy_name=None,seed=None,accuracy=1.0,heuristic=None,heuristic_param=None):

    global global_heuristic, global_heuristic_param,min_acc
    global_heuristic=heuristic
    global_heuristic_param=heuristic_param
    min_acc=1/max_episode_length
    

    # Inicialización de contadores.
    global n_steps_proc,n_steps_acc,n_episodes,n_generations
    n_steps_proc=0 # Contador que índica el número de steps consumidos hasta el momento para el procedimiento.
    n_steps_acc=0 # Contador que índica el número de steps consumidos hasta el momento para el ajuste del accuracy.
    n_episodes = 0 # Contador que indica en que episodio estamos.
    n_generations=0 # Contador que indica en que generación estamos.

    # Definición de parámetros para el algoritmos CMA-ES.
    global popsize,batch_size_ep,total_generations
    popsize= 100 # Tamaño de las generaciones/poblaciones en CMA-ES.
    batch_size_ep=1 # Número de episodios que se van a considerar para evaluar cada política/individuos de una generación.

    # Fijar semilla.
    set_seed(seed)

    with TFTrainer(ctxt) as trainer:

        # Definir entorno de entrenamiento con el accuracy seleccionado.
        global current_max_episode_length
        current_max_episode_length=max_episode_length*accuracy
        train_env = GymEnv(gymEnvName, max_episode_length=current_max_episode_length)
        train_env._env.unwrapped.model.opt.timestep=default_frametime/accuracy
        train_env._env.seed(seed=train_env_seed)

        # Definir entorno de validación con el accuracy máximo.
        global test_env
        test_env = GymEnv(gymEnvName, max_episode_length=max_episode_length)
        test_env._env.unwrapped.model.opt.timestep=default_frametime
        
        # Definir política.
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
        total_generations= int(max_steps/(popsize*batch_size_ep*current_max_episode_length)) # Número de generaciones a evaluar, dependiendo del número máximo de steps definido para entrenar. (CRITERIO DE PARADA)
        trainer.train(n_epochs=total_generations, batch_size=batch_size_ep*current_max_episode_length)

#--------------------------------------------------------------------------------------------------
# Funciones auxiliares para definir el accuracy apropiado en cada momento del proceso.
#--------------------------------------------------------------------------------------------------
# FUNCIÓN (Cálculo de la correlación de Spearman entre dos secuencias)
def spearman_corr(x,y):
    return sc.stats.spearmanr(x,y)[0]

# FUNCIÓN (Convertir vector de rewards en vector de ranking)
def from_rewards_to_ranking(list_rewards):
    list_pos_ranking=np.argsort(np.array(list_rewards))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

def get_environment_with_accuracy(accuracy):

    env = GymEnv(gymEnvName, max_episode_length=max_episode_length*accuracy)
    env._env.unwrapped.model.opt.timestep=default_frametime/accuracy
    env._env.seed(seed=train_env_seed)

    return env

# FUNCIÓN  (Generar lista con los rewards asociados a cada política que forma la generación)
def generation_reward_list(population,accuracy,count_time_acc=True,count_time_gen=False):

    global batch_size_ep,n_steps_proc,n_steps_acc

    # Generar entorno con accuracy apropiado.
    env=get_environment_with_accuracy(accuracy)

    # Definir número de episodios a evaluar.
    if count_time_gen:
        n_eval_episodes=test_n_eval_episodes
    else:
        n_eval_episodes=batch_size_ep
    
    # Evaluar población.
    list_rewards=[]
    for policy in population:
        reward=evaluate_policy(policy,env=env,n_eval_episodes=n_eval_episodes,type_eval_test=count_time_gen)
        if count_time_acc and not count_time_gen:
            n_steps_acc+=len(population)*batch_size_ep*(max_episode_length/accuracy)
        if count_time_gen:
            n_steps_proc+=len(population)*test_n_eval_episodes*(max_episode_length/accuracy)
        list_rewards.append(reward)

    return list_rewards
#--------------------------------------------------------------------------------------------------
# Funciones asociadas a los heurísticos que se aplicarán para ajustar el accuracy.
#--------------------------------------------------------------------------------------------------

# FUNCIÓN  (Implementación adaptada del método de bisección)
def bisection_method(init_acc,population,train_seed,threshold=0.95):

    # Inicializar límite inferior y superior.
    acc0=init_acc
    acc1=1    

    # Punto intermedio.
    prev_m=init_acc
    m=(acc0+acc1)/2
    
    # Función para calcular la correlación entre los rankings del 10% aleatorio de las políticas
    # usando el accuracy actual y el máximo.
    def similarity_between_current_best_acc(acc,population,train_seed,first_iteration):

        # Seleccionar de forma aleatoria el 10% de las políticas que forman la generación.
        random.seed(train_seed)
        ind_sol=random.sample(range(len(population)),int(len(population)*0.1))
        list_solutions=list(np.array(population)[ind_sol])

        # Guardar los rewards asociados a cada solución seleccionada.
        best_rewards=generation_reward_list(list_solutions,1,count_time_acc=first_iteration)# Con el máximo accuracy. 
        new_rewards=generation_reward_list(list_solutions,acc)# Accuracy nuevo. 

        # Obtener vectores de rankings asociados.
        new_ranking=from_rewards_to_ranking(new_rewards)# Accuracy nuevo. 
        best_ranking=from_rewards_to_ranking(best_rewards)# Máximo accuracy. 
                
        # Comparar ambos rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)


        return metric_value

    # Reajustar límites del intervalo hasta que este tenga un rango lo suficientemente pequeño.
    first_iteration=True
    while acc1-acc0>0.1:
        metric_value=similarity_between_current_best_acc(m,population,train_seed,first_iteration)
        if metric_value>=threshold:
            acc1=m
        else:
            acc0=m

        prev_m=m
        m=(acc0+acc1)/2
        
        first_iteration=False

    return prev_m

# FUNCIÓN (Ejecutar heurísticos durante el proceso de entrenamiento)
def execute_heuristic(gen,min_acc,acc,population,train_seed,list_variances,heuristic,param):
    global last_optimal_time
    global n_steps_proc
    global n_steps_acc
    global max_steps
    global batch_size_ep

    # HEURÍSTICO 7 de Symbolic Regressor: Bisección de generación en generación (el umbral es el parámetro).
    if heuristic==7: 
        acc=bisection_method(min_acc,population,train_seed,threshold=param)

    # HEURÍSTICO 9 de Symbolic Regressor: Bisección con frecuencia constante (parámetro) 
    # de actualización de accuracy y umbral de método de bisección fijado en 0.95.
    if heuristic==9: 
        if gen==0:
            acc=bisection_method(min_acc,population,train_seed)
            last_optimal_time=n_steps_proc+n_steps_acc

        else:
            if (n_steps_proc+n_steps_acc)-last_optimal_time>=int(max_steps*param):
                acc=bisection_method(min_acc,population,train_seed)
                last_optimal_time=n_steps_proc+n_steps_acc

    # HEURÍSTICO 12 de Symbolic Regressor: Bisección con definición automática para frecuencia 
    # de actualización de accuracy (depende de parámetro) y umbral del método de bisección fijado en 0.95.
    if heuristic==12: 
        if gen==0: 
            acc=bisection_method(min_acc,population,train_seed)
            
        else:
            if len(list_variances)>=param+1:
                # Función para calcular el intervalo de confianza.
                def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                    mean_list=[]
                    for i in range(bootstrap_iterations):
                        sample = np.random.choice(data, len(data), replace=True) 
                        mean_list.append(np.mean(sample))
                    return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

                variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-param):-2])
                last_variance=list_variances[-1]

                # Calcular el mínimo accuracy con el que se obtiene la máxima calidad.
                if last_variance<variance_q05 or last_variance>variance_q95:
                    acc=bisection_method(min_acc,population,train_seed)

    
    return acc
    

#==================================================================================================
# FUNCIONES DISEÑADAS PARA SUSTITUIR ALGUNAS YA EXISTENTES
#==================================================================================================
def train(self, trainer):

    init_mean = self.policy.get_param_values()
    self._es = cma.CMAEvolutionStrategy(init_mean, self._sigma0,
                                        {'popsize': self._n_samples})
    self._all_params = self._sample_params()
    self._cur_params = self._all_params[0]
    self.policy.set_param_values(self._cur_params)
    self._all_returns = []

    # start actual training
    last_return = None
    global n_steps_proc,n_steps_acc
    while n_steps_proc+n_steps_acc<max_steps:# MODIFICACIÓN: cambiar criterio de parada.
        # MODIFICACIÓN: primero obtener las todas las políticas que forman la generación.
        global list_gen_policies
        list_gen_policies=[]
        for _ in range(self._n_samples):
            policy = getattr(self._algo, 'exploration_policy', None)
            if policy is None:
                policy = trainer._algo.policy

            list_gen_policies.append(policy)

        # MODIFICACIÓN: ajustar el accuracy en el entorno.
        global n_generations,accuracy,seed,df,min_acc,global_heuristic,global_heuristic_param,optimal_acc
        df_seed=pd.DataFrame(df)
        df_seed=df_seed[df_seed[1]==seed]
        optimal_acc=execute_heuristic(n_generations,min_acc,accuracy,list_gen_policies,seed,list(df_seed[5]),global_heuristic,global_heuristic_param)
        train_env = GymEnv(gymEnvName, max_episode_length=max_episode_length*optimal_acc)
        train_env._env.unwrapped.model.opt.timestep=default_frametime/optimal_acc
        train_env._env.seed(seed=train_env_seed)
        trainer._env=train_env

        global idx_generation_policy
        for idx_generation_policy in range(self._n_samples):
            trainer.step_episode = trainer.obtain_episodes(trainer.step_itr)
            last_return = self._train_once(trainer.step_itr,trainer.step_episode)
            trainer.step_itr += 1

    return last_return


def obtain_episodes(self,itr, batch_size=None,agent_update=None, env_update=None):

    if self._sampler is None:
        raise ValueError('trainer was not initialized with `sampler`. '
                            'the algo should have a `_sampler` field when'
                            '`setup()` is called')
    if batch_size is None and self._train_args.batch_size is None:
        raise ValueError(
            'trainer was not initialized with `batch_size`. '
            'Either provide `batch_size` to trainer.train, '
            ' or pass `batch_size` to trainer.obtain_samples.')
    episodes = None
    # if agent_update is None:
    #     policy = getattr(self._algo, 'exploration_policy', None)
    #     if policy is None:
    #         # This field should exist, since self.make_sampler would have
    #         # failed otherwise.
    #         policy = self._algo.policy
    #     agent_update = policy.get_param_values()

    # MODIFICACIÓN: acceder a datos almacenados en las listas en train() de arriba.
    policy=list_gen_policies[idx_generation_policy]
    agent_update=policy.get_param_values()

    episodes = self._sampler.obtain_samples(itr, (batch_size or self._train_args.batch_size),agent_update=agent_update,env_update=env_update)
    self._stats.total_env_steps += sum(episodes.lengths)
    return episodes

# FUNCIÓN (para poder evaluar cada política de cada generación durante el entrenamiento con el mismo
# conjunto de episodios, esta modificación hace que el proceso sea determinista y las comparaciones 
# de individuos por generación sean justas).
def start_episode(self):

    self._eps_length = 0

    # MODIFICACIÓN: para que siempre se usen los mismos episodios para validar las políticas/individuos y estas puedan ser comparables.
    global n_episodes,batch_size_ep,n_generations
    if n_episodes%batch_size_ep==0:
        self.env._env.seed(seed=n_generations)#seed=train_env_seed (si se quiere evaluar los mismos episodios en todas las generaciones)
    self._prev_obs, episode_info = self.env.reset()

    for k, v in episode_info.items():
        self._episode_infos[k].append(v)

    self.agent.reset()

# FUNCIÓN  (para poder almacenar los datos de interés durante el entrenamiento).
def rollout(self):
    global n_steps_proc,n_steps_acc,n_episodes,n_generations,df,current_max_episode_length,test_n_eval_episodes,popsize,total_generations
    global seed,global_heuristic_param,optimal_acc
    global list_gen_policies,list_gen_policies_rewards,policy_reward_per_ep

    # Listas para guardar las políticas asociadas a una generación y sus correspondientes rewards.
    n_policies=n_episodes/batch_size_ep #Políticas evaluadas hasta el momento.
    if n_policies%popsize==0:

        list_gen_policies=[]
        list_gen_policies_rewards=[]

        policy_reward_per_ep=[]

    # En caso de usar un entorno con criterio de parada dependiente del parámetro 
    # terminate_when_unhealthy, anular este tipo de parada.
    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    # Inicializar episodio.
    self.start_episode() 
    self._max_episode_length = current_max_episode_length # Fijar tamaño máximo del episodio.
    
    # Actualizar contadores tras evaluar el episodio.
    episode_steps = 0 # Inicializar contador para sumar el número de steps dados en este episodio.
    policy_reward=0
    while not self.step_episode(): # Hasta que no se puedan dar más steps en el episodio.
        policy_reward+=self._env_steps[episode_steps].reward # Dar un nuevo step.
        episode_steps+= 1 # Sumar step.

    n_steps_proc += episode_steps #  Steps totales dados hasta el momento.
    n_episodes += 1 # Episodios evaluados hasta el momento.
    policy_reward_per_ep.append(policy_reward)

    # Cuando se haya evaluado una política/individuo al completo (número de episodios evaluado=batch_size_ep),
    # se almacena la política y su correspondiente reward (reward total en batch_size_ep).
    if n_episodes%batch_size_ep==0:
        list_gen_policies.append(self.agent)
        list_gen_policies_rewards.append(np.mean(policy_reward_per_ep))
        policy_reward_per_ep=[]
    
    # Cuando se han evaluado las suficientes políticas como para poder completar una generación nueva,
    # se guarda la información obtenida de la nueva generación.
    n_policies=n_episodes/batch_size_ep#Políticas evaluadas hasta el momento.
    if n_policies!=0 and n_policies%popsize == 0:
        
        # Restar número de steps sumados de forma duplicada (en la última iteración del método de bisección ya se han evaluado 
        # el 10% de las políticas que forman la generación, y los steps dados se han vuelto a contar ahora en n_steps_proc)
        n_steps_acc-=int(popsize*0.1)*batch_size_ep*(max_episode_length*optimal_acc)

        # Actualizar base de datos.
        n_generations+=1
        best_policy=list_gen_policies[list_gen_policies_rewards.index(max(list_gen_policies_rewards))]
        reward=evaluate_policy(best_policy,test_env,test_n_eval_episodes)
        df.append([global_heuristic_param,seed,n_generations,reward,optimal_acc,np.var(list_gen_policies_rewards),n_steps_proc,n_steps_acc,n_steps_proc+n_steps_acc])
        

        # print('GENERACIÓN: '+str(n_generations)+'/'+str(total_generations))

    return self.collect_episode()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Modificación de función existente (para guardar datos durante el entrenamiento).
garage.sampler.default_worker.DefaultWorker.rollout = rollout
# Modificación de funciones existentes (para poder aplicar los heurísticos durante el entrenamiento)
CMAES.train=train
Trainer.obtain_episodes=obtain_episodes
# Modificación de función existente ( para no perder tiempo).
garage.trainer.Trainer.save = lambda self, epoch: "skipp save."
# Modificación de función existente (para no imprimir información de episodio durante el entrenamiento).
Logger.log= lambda self, data: 'skipp info message.'
warnings.filterwarnings("ignore")
# Modificación de función existente (para evaluar todas las políticas/individuos de cada generación con los mismos episodios y así puedan ser comparables).
from garage.sampler.default_worker import DefaultWorker
DefaultWorker.start_episode=start_episode


# Características del entorno.
gymEnvName='Swimmer-v3'
action_space="continuous"
max_episode_length=1000
default_frametime=0.01 # Parámetro del que se modificará el accuracy.
policy_name='SwimmerPolicy'
DTU=False # Criterio de parada dependiente de terminate_when_unhealthy.

# Mallados y parámetros para el entrenamiento.
list_train_seeds = list(range(2,32,1)) # Lista de semillas de entrenamiento.
list_acc=[1.0,0.8,0.6,0.4]#[0.5,0.3,0.2,0.1]
train_env_seed=0 # Semilla para el entorno de entrenamiento.
max_steps=1000000 # Límite de entrenamiento medido en steps (asumiendo que el coste de todos los steps es el mismo). (Referencia: https://huggingface.co/sb3/ppo-Swimmer-v3/tree/main)

# Parámetros de validación.
test_env_seed=1 # Semilla para el entorno de validación.
test_n_eval_episodes=10 # Número de episodios que se evaluarán en la validación.

# Lista de argumentos para el procesamiento en paralelo.
list_arg=[[7,0.8],[7,0.95],[9,0.1],[9,0.3],[12,5],[12,10]]


# Función para ejecución en paralelo.
def parallel_processing(arg):

    heuristic=arg[0]
    heuristic_param=arg[1]

    global df
    df=[]

    global seed,accuracy
    accuracy=arg
    for seed in tqdm(list_train_seeds):
        learn(gymEnvName=gymEnvName, action_space=action_space, max_episode_length=max_episode_length,policy_name=policy_name,seed=seed,accuracy=accuracy,heuristic=heuristic,heuristic_param=heuristic_param) 

    # Guardar base de datos.
    df=pd.DataFrame(df,columns=['heuristic_param','seed','n_gen','score','accuracy','variance','n_steps_proc','n_steps_acc','n_steps'])
    df.to_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(heuristic_param)+'.csv')

# Procesamiento en paralelo.
phisical_cpu=ps.cpu_count(logical=False)
pool=mp.Pool(phisical_cpu)
pool.map(parallel_processing,list_acc)
pool.close()







